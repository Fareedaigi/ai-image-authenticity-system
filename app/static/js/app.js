(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const drop = $("drop"), input = $("file");
  let lastReport = null, previewUrl = null;

  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
  const fmtBytes = (b) => b > 1 << 20 ? (b / (1 << 20)).toFixed(1) + " MB" : (b / 1024).toFixed(0) + " KB";
  const show = (el, on) => { el.hidden = !on; };

  // ---- input handling ---------------------------------------------------
  drop.addEventListener("click", () => input.click());
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); }
  });
  input.addEventListener("change", () => input.files[0] && analyse(input.files[0]));
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => {
    e.preventDefault(); drop.classList.add("over");
  }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => {
    e.preventDefault(); drop.classList.remove("over");
  }));
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) analyse(f);
  });
  window.addEventListener("paste", (e) => {
    const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
    if (item) analyse(item.getAsFile());
  });
  $("again").addEventListener("click", () => {
    show($("results"), false);
    input.value = "";
    window.scrollTo({ top: 0, behavior: "smooth" });
    drop.focus();
  });
  $("download").addEventListener("click", () => {
    if (!lastReport) return;
    const slim = structuredClone(lastReport);
    slim.analyzers.forEach((a) => delete a.visual);
    const blob = new Blob([JSON.stringify(slim, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = (lastReport.filename || "image").replace(/\.[^.]+$/, "") + "-authenticity.json";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  // ---- analysis -----------------------------------------------------------
  async function analyse(file) {
    show($("error"), false);
    show($("results"), false);
    show($("status"), true);
    $("status-text").textContent = `Analysing ${file.name || "image"}…`;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);

    const body = new FormData();
    body.append("image", file, file.name || "pasted.png");
    try {
      const res = await fetch("/api/analyze", { method: "POST", body });
      const data = await res.json().catch(() => ({ error: `Server error (${res.status})` }));
      if (!res.ok) throw new Error(data.error || `Server error (${res.status})`);
      lastReport = data;
      render(data);
    } catch (err) {
      $("error").textContent = err.message;
      show($("error"), true);
    } finally {
      show($("status"), false);
    }
  }

  // ---- rendering ------------------------------------------------------------
  const KIND_ICON = { ai: "!", real: "✓", info: "•" };

  function render(r) {
    const s = r.summary;
    $("preview").src = previewUrl;
    $("file-info").innerHTML = [
      ["File", r.filename],
      ["Format", r.image.format],
      ["Size", `${r.image.width} × ${r.image.height} · ${fmtBytes(r.image.bytes)}`],
      ["Analysis time", `${(r.runtime_ms / 1000).toFixed(2)} s`],
    ].map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");

    // score + gauge
    const card = document.querySelector(".score-card");
    card.className = `card score-card tone-${s.tone}`;
    $("score").textContent = s.authenticity_score;
    const col = s.tone.startsWith("ai") ? "var(--ai)" : s.tone.startsWith("real") ? "var(--real)" : "var(--warn)";
    const fg = $("gauge-fg");
    fg.style.stroke = col;
    fg.style.strokeDashoffset = 326.7;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      fg.style.strokeDashoffset = 326.7 * (1 - s.authenticity_score / 100);
    }));
    $("verdict").textContent = s.verdict;
    $("ai-prob").textContent = `AI likelihood ${(s.ai_probability * 100).toFixed(1)}%`;
    $("confidence").textContent = s.confidence;

    // attribution
    const at = s.attribution, atEl = $("attribution");
    if (at.platform && at.basis === "metadata") {
      const ev = at.candidates[0].evidence.slice(0, 3).map(esc).join(" · ");
      const others = at.candidates.slice(1).map((c) => esc(c.platform)).join(", ");
      atEl.className = "attribution found";
      atEl.innerHTML = `<span class="muted">Likely source (from metadata)</span>
        <strong>${esc(at.platform)}</strong><span class="muted">${ev}</span>
        ${others ? `<br><span class="muted">Also referenced: ${others}</span>` : ""}
        ${at.note ? `<br><span class="muted">${esc(at.note)}</span>` : ""}`;
    } else if (at.platform) {
      atEl.className = "attribution";
      atEl.innerHTML = `<span class="muted">Possible source (weak — output size only)</span>
        <strong>${esc(at.platform)}</strong>`;
    } else {
      atEl.className = "attribution";
      atEl.innerHTML = `<span class="muted">Source platform</span><strong>Not identifiable</strong>
        <span class="muted">No provenance metadata names a generator. Pixel analysis can estimate
        whether an image is AI-made, but not reliably which platform made it.</span>`;
    }

    $("reasons").innerHTML = s.key_reasons.map((k) =>
      `<li><span>${esc(k.text)} <span class="tag">— ${esc(k.analyzer)}</span></span></li>`).join("")
      || `<li><span>No single strong indicator; the result rests on weak signals.</span></li>`;

    // contributions
    const maxAbs = Math.max(2, ...s.contributions.map((c) => Math.abs(c.contribution || 0)));
    $("contrib").innerHTML = s.contributions.map((c) => {
      if (c.contribution === null) {
        return `<div class="crow"><span class="cname">${esc(c.name)}</span><span class="na">not applicable</span><span></span></div>`;
      }
      const w = Math.min(50, (Math.abs(c.contribution) / maxAbs) * 50);
      const cls = c.contribution >= 0 ? "ai" : "real";
      return `<div class="crow"><span class="cname">${esc(c.name)}</span>
        <div class="cbar" role="img" aria-label="${esc(c.name)} contribution ${c.contribution}">
          <div class="cfill ${cls}" style="width:${w}%"></div></div>
        <span class="cval">${c.contribution > 0 ? "+" : ""}${c.contribution.toFixed(2)}</span></div>`;
    }).join("");

    // analyzer cards
    $("analyzers").innerHTML = r.analyzers.map(analyzerCard).join("");
    $("disclaimer").textContent = r.disclaimer;
    show($("results"), true);
    $("results").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function pill(a) {
    if (!a.applicable) return `<span class="pill">N/A</span>`;
    const p = a.ai_probability;
    const cls = p >= 0.6 ? "ai" : p <= 0.4 ? "real" : "neutral";
    const label = p >= 0.6 ? "Suspicious" : p <= 0.4 ? "Looks authentic" : "Neutral";
    return `<span class="pill ${cls}">${label} · ${(p * 100).toFixed(0)}% AI</span>`;
  }

  function detailRows(obj, prefix = "") {
    return Object.entries(obj).flatMap(([k, v]) => {
      if (v === null || v === undefined || (Array.isArray(v) && !v.length) || k === "runtime_ms") return [];
      if (k === "platform_candidates") {
        return v.map((c) => `<tr><td>platform: ${esc(c.platform)}</td><td>${esc(c.evidence.join("; "))}</td></tr>`);
      }
      if (k === "faces") {
        return v.map((f, i) => `<tr><td>face ${i + 1}</td><td>${esc(Object.entries(f)
          .filter(([kk]) => kk !== "notes").map(([kk, vv]) => `${kk}: ${vv}`).join(", "))}</td></tr>`);
      }
      if (typeof v === "object" && !Array.isArray(v)) return detailRows(v, prefix + k + " › ");
      const val = Array.isArray(v) ? v.join(", ") : typeof v === "boolean" ? (v ? "yes" : "no") : v;
      return [`<tr><td>${esc(prefix + k.replaceAll("_", " "))}</td><td>${esc(val)}</td></tr>`];
    });
  }

  function analyzerCard(a) {
    const findings = a.findings.map((f) =>
      `<li class="${f.kind}"><span class="ic">${KIND_ICON[f.kind] || "•"}</span><span>${esc(f.text)}</span></li>`).join("");
    const visual = a.visual ? `<figure class="visual"><img src="${a.visual}" alt="${esc(a.name)} diagnostic image" loading="lazy">
      <figcaption>${esc(a.visual_caption || "")}</figcaption></figure>` : "";
    const rows = detailRows(a.details);
    const details = rows.length ? `<details><summary>Technical details</summary>
      <table class="kv">${rows.join("")}</table></details>` : "";
    return `<article class="card analyzer">
      <div class="an-head"><h2>${esc(a.name)}</h2>${pill(a)}</div>
      <p class="an-desc">${esc(a.description)}</p>
      <ul class="findings">${findings}</ul>${visual}${details}</article>`;
  }
})();
