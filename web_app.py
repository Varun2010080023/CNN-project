"""Serve the standalone Fruit Lens interface and local CNN predictions."""

import io
import json
import warnings
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image, UnidentifiedImageError

from model import MODEL_PATH, ROOT, load_model, predict


WEB_ROOT = ROOT / "web"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/fruit-backdrop.svg": ("fruit-backdrop.svg", "image/svg+xml"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}


@lru_cache(maxsize=1)
def cached_model(modified_at):
    """Reload the checkpoint when a retraining run replaces it."""
    return load_model()


class FruitLensHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        asset = ASSETS.get(self.path.split("?", 1)[0])
        if asset is None:
            self.send_error(404)
            return
        filename, content_type = asset
        content = (WEB_ROOT / filename).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_json(400, {"error": "Invalid upload size."})
            return
        if size < 1 or size > MAX_UPLOAD_BYTES:
            self.send_json(413 if size > MAX_UPLOAD_BYTES else 400,
                           {"error": "Choose an image smaller than 10 MB."})
            return
        image_bytes = self.rfile.read(size)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(image_bytes)) as source:
                    if source.format not in ("JPEG", "PNG", "WEBP"):
                        raise ValueError("Unsupported image format.")
                    photo = source.copy()
        except (UnidentifiedImageError, OSError, ValueError,
                Image.DecompressionBombError, Image.DecompressionBombWarning):
            self.send_json(400, {"error": "That file could not be read as a JPG, PNG, or WebP image."})
            return

        if not MODEL_PATH.exists():
            self.send_json(503, {"error": "The model is not trained yet. Run python train.py first."})
            return
        try:
            fruit, probabilities = predict(cached_model(MODEL_PATH.stat().st_mtime_ns), photo)
            fruit_info = json.loads((ROOT / "fruit_info.json").read_text(encoding="utf-8"))[fruit]
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, EOFError) as error:
            self.send_json(500, {"error": f"Prediction is unavailable: {error}"})
            return
        self.send_json(200, {
            "fruit": fruit,
            "probabilities": probabilities,
            "info": fruit_info,
        })

    def send_json(self, status, payload):
        content = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run the standalone Fruit Lens web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    torch.set_num_threads(2)
    server = ThreadingHTTPServer((args.host, args.port), FruitLensHandler)
    print(f"Fruit Lens is ready at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
