from flask import Flask

MAX_UPLOAD_MB = 25


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

    from .routes import bp
    app.register_blueprint(bp)
    return app
