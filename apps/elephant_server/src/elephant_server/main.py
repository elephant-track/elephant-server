from flask import Flask


def create_app():
    app = Flask(__name__)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def main():
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
