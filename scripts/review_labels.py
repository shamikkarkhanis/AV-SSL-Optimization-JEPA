from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import threading
import urllib.parse
from io import BytesIO
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def infer_manifest_path(labels_path: Path) -> Path:
    stem = labels_path.stem
    if stem.endswith("_evaluation_labels"):
        base = stem[: -len("_evaluation_labels")]
        for suffix in (".jsonl", ".py"):
            candidate = labels_path.with_name(f"{base}{suffix}")
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"Could not infer manifest path for {labels_path}. Pass --manifest-path explicitly."
    )


def _infer_data_root(manifest_records: List[Dict[str, Any]], repo_root: Path) -> Optional[Path]:
    source_dataset = (
        manifest_records[0].get("metadata", {}).get("source_dataset")
        if manifest_records
        else None
    )
    if source_dataset:
        candidate = repo_root / "data" / "raw" / str(source_dataset)
        if candidate.exists():
            return candidate
        backup_candidate = repo_root / "data" / "raw.bak" / "v1.0-mini"
        if str(source_dataset) == "file" and backup_candidate.exists():
            return backup_candidate
        backup_candidate = repo_root / "data" / "raw.bak" / str(source_dataset)
        if backup_candidate.exists():
            return backup_candidate
    return None


def _resolve_frame_path(frame_path: str, data_root: Optional[Path], manifest_path: Path) -> Path:
    candidate = Path(frame_path)
    if candidate.is_absolute():
        return candidate
    if data_root is not None:
        variants = [
            data_root / candidate,
            data_root / "samples" / candidate,
            data_root / candidate.name,
        ]
        for variant in variants:
            if variant.exists():
                return variant
        return variants[0]
    return manifest_path.parent / candidate


def _sample_frame_paths(frame_paths: List[str], max_frames: int = 4) -> List[str]:
    if len(frame_paths) <= max_frames:
        return frame_paths
    last_index = len(frame_paths) - 1
    indices = sorted({round(i * last_index / (max_frames - 1)) for i in range(max_frames)})
    return [frame_paths[index] for index in indices]


def _build_clip_gif_bytes(
    frame_paths: List[str],
    data_root: Optional[Path],
    manifest_path: Path,
    max_side: int = 640,
    duration_ms: int = 120,
) -> bytes:
    frames: List[Image.Image] = []
    for frame_path in frame_paths:
        resolved = _resolve_frame_path(frame_path, data_root, manifest_path)
        if not resolved.exists():
            continue
        with Image.open(resolved) as image:
            frame = image.convert("RGB")
            frame.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            frames.append(frame.copy())
    if not frames:
        raise FileNotFoundError("No clip frames could be resolved for GIF rendering.")

    buffer = BytesIO()
    frames[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return buffer.getvalue()


def update_label_file(
    labels_path: Path,
    rows: List[Dict[str, Any]],
    clip_id: str,
    binary_label: Optional[int],
) -> None:
    updated = False
    for row in rows:
        if str(row.get("clip_id")) == clip_id:
            row["binary_label"] = binary_label
            updated = True
            break
    if not updated:
        raise KeyError(f"Clip id {clip_id} not found in labels file.")

    tmp_path = labels_path.with_suffix(f"{labels_path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    shutil.move(str(tmp_path), str(labels_path))


@dataclass
class ClipReviewRow:
    clip_id: str
    split: str
    scene_id: str
    camera: str
    binary_label: Optional[int]
    frame_paths: List[str]


class ReviewStore:
    def __init__(
        self,
        labels_path: Path,
        manifest_path: Path,
        data_root: Optional[Path] = None,
    ) -> None:
        self.labels_path = labels_path
        self.manifest_path = manifest_path
        self.label_rows = _read_jsonl(labels_path)
        self.manifest_rows = _read_jsonl(manifest_path)
        if data_root is None:
            data_root = _infer_data_root(self.manifest_rows, Path(__file__).resolve().parent.parent)
        self.data_root = data_root
        self._lock = threading.Lock()
        self._clip_gif_cache: Dict[str, bytes] = {}

        manifest_by_clip = {
            str(row["clip_id"]): row for row in self.manifest_rows if row.get("clip_id") is not None
        }
        self.clips: List[ClipReviewRow] = []
        for row in self.label_rows:
            clip_id = str(row["clip_id"])
            manifest_row = manifest_by_clip.get(clip_id)
            if manifest_row is None:
                raise KeyError(f"Clip id {clip_id} missing from manifest {manifest_path}.")
            self.clips.append(
                ClipReviewRow(
                    clip_id=clip_id,
                    split=str(row.get("split") or manifest_row.get("split") or ""),
                    scene_id=str(row.get("scene_id") or manifest_row.get("scene_id") or ""),
                    camera=str(row.get("camera") or manifest_row.get("camera") or ""),
                    binary_label=row.get("binary_label"),
                    frame_paths=list(manifest_row.get("frame_paths") or []),
                )
            )

    def counts(self) -> Dict[str, int]:
        positives = sum(clip.binary_label == 1 for clip in self.clips)
        negatives = sum(clip.binary_label == 0 for clip in self.clips)
        unlabeled = sum(clip.binary_label is None for clip in self.clips)
        return {
            "total": len(self.clips),
            "positives": positives,
            "negatives": negatives,
            "unlabeled": unlabeled,
        }

    def clip_payload(self, index: int) -> Dict[str, Any]:
        clip = self.clips[index]
        frames = []
        for frame_path in _sample_frame_paths(clip.frame_paths):
            resolved = _resolve_frame_path(frame_path, self.data_root, self.manifest_path)
            if resolved.exists():
                frames.append(
                    {
                        "display_path": frame_path,
                        "url": f"/frame?path={urllib.parse.quote(str(resolved))}",
                    }
                )
            else:
                frames.append(
                    {
                        "display_path": frame_path,
                        "missing": True,
                    }
                )
        return {
            "index": index,
            "clip_id": clip.clip_id,
            "split": clip.split,
            "scene_id": clip.scene_id,
            "camera": clip.camera,
            "binary_label": clip.binary_label,
            "counts": self.counts(),
            "clip_url": f"/clip?clip_id={urllib.parse.quote(clip.clip_id)}",
            "frames": frames,
        }

    def first_unlabeled_index(self) -> int:
        for index, clip in enumerate(self.clips):
            if clip.binary_label is None:
                return index
        return 0

    def set_label(self, index: int, binary_label: Optional[int]) -> Dict[str, Any]:
        with self._lock:
            clip = self.clips[index]
            clip.binary_label = binary_label
            update_label_file(self.labels_path, self.label_rows, clip.clip_id, binary_label)
            return self.clip_payload(index)

    def clip_gif_bytes(self, clip_id: str) -> bytes:
        with self._lock:
            cached = self._clip_gif_cache.get(clip_id)
            if cached is not None:
                return cached
            clip = next((item for item in self.clips if item.clip_id == clip_id), None)
            if clip is None:
                raise KeyError(f"Clip id {clip_id} not found.")
            gif_bytes = _build_clip_gif_bytes(
                clip.frame_paths,
                data_root=self.data_root,
                manifest_path=self.manifest_path,
            )
            self._clip_gif_cache[clip_id] = gif_bytes
            return gif_bytes


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Clip Label Review</title>
  <style>
    :root {
      --bg: #f4efe6;
      --panel: #fffaf1;
      --ink: #1c1917;
      --muted: #6b6257;
      --accent: #b45309;
      --good: #166534;
      --bad: #991b1b;
      --border: #d6c7ad;
    }
    body {
      margin: 0;
      font-family: Georgia, "Iowan Old Style", serif;
      background: radial-gradient(circle at top, #fffaf1 0%, var(--bg) 55%, #eadfcf 100%);
      color: var(--ink);
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 24px;
    }
    .panel {
      background: rgba(255, 250, 241, 0.94);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px 20px;
      box-shadow: 0 18px 50px rgba(60, 34, 9, 0.08);
      backdrop-filter: blur(8px);
    }
    .header {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: start;
      margin-bottom: 16px;
    }
    h1 {
      margin: 0 0 6px 0;
      font-size: 32px;
      line-height: 1.05;
    }
    .meta, .status, .help {
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
      line-height: 1.6;
      white-space: pre-wrap;
    }
    .status strong {
      color: var(--ink);
    }
    .frames {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px;
      margin-top: 18px;
    }
    .player {
      margin-top: 18px;
      background: #efe5d4;
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    .player img {
      width: 100%;
      max-height: 540px;
      object-fit: contain;
      background: #d9c7aa;
    }
    .player .caption {
      padding: 10px 12px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      color: var(--muted);
    }
    figure {
      margin: 0;
      background: #efe5d4;
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    img {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: cover;
      background: #d9c7aa;
    }
    figcaption {
      padding: 10px 12px;
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      color: var(--muted);
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }
    button {
      border: 0;
      border-radius: 999px;
      padding: 12px 16px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      color: white;
    }
    .good { background: var(--good); }
    .bad { background: var(--bad); }
    .clear { background: #57534e; }
    .nav { background: var(--accent); }
    .active {
      outline: 3px solid rgba(180, 83, 9, 0.28);
      outline-offset: 2px;
    }
  </style>
</head>
<body>
  <main>
    <div class="panel">
      <div class="header">
        <div>
          <h1 id="title">Clip review</h1>
          <div class="meta" id="meta"></div>
        </div>
        <div class="status" id="status"></div>
      </div>
      <div class="help">Keys: 1 = positive, 0 = negative, U = clear, J = next, K = previous</div>
      <div class="actions">
        <button class="good" id="positive">Label 1</button>
        <button class="bad" id="negative">Label 0</button>
        <button class="clear" id="clear">Clear</button>
        <button class="nav" id="prev">Prev</button>
        <button class="nav" id="next">Next</button>
      </div>
      <div class="player">
        <img id="clip-player" alt="Rendered clip playback">
        <div class="caption" id="clip-caption"></div>
      </div>
      <div class="frames" id="frames"></div>
    </div>
  </main>
  <script>
    let currentIndex = 0;

    async function fetchClip(index) {
      const response = await fetch(`/api/clip?index=${index}`);
      if (!response.ok) {
        throw new Error(await response.text());
      }
      return response.json();
    }

    async function setLabel(value) {
      const response = await fetch("/api/label", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({index: currentIndex, binary_label: value}),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const clip = await response.json();
      renderClip(clip);
    }

    function renderClip(clip) {
      currentIndex = clip.index;
      document.getElementById("title").textContent = `Clip ${clip.index + 1} / ${clip.counts.total}`;
      document.getElementById("meta").textContent =
        `clip_id: ${clip.clip_id}\nscene: ${clip.scene_id}\nsplit: ${clip.split}\ncamera: ${clip.camera}`;
      document.getElementById("status").innerHTML =
        `<strong>current</strong>: ${clip.binary_label === null ? "unlabeled" : clip.binary_label}` +
        `\n<strong>counts</strong>: +${clip.counts.positives} / -${clip.counts.negatives} / ?${clip.counts.unlabeled}`;

      document.getElementById("positive").classList.toggle("active", clip.binary_label === 1);
      document.getElementById("negative").classList.toggle("active", clip.binary_label === 0);
      document.getElementById("clear").classList.toggle("active", clip.binary_label === null);

      const player = document.getElementById("clip-player");
      player.src = `${clip.clip_url}&t=${Date.now()}`;
      document.getElementById("clip-caption").textContent = "Autoplay loop rendered from clip frames";

      const frames = document.getElementById("frames");
      frames.innerHTML = "";
      for (const frame of clip.frames) {
        const figure = document.createElement("figure");
        const img = document.createElement("img");
        if (frame.url) {
          img.src = frame.url;
        }
        figure.appendChild(img);
        const caption = document.createElement("figcaption");
        caption.textContent = frame.missing ? `${frame.display_path} (missing)` : frame.display_path;
        figure.appendChild(caption);
        frames.appendChild(figure);
      }
    }

    async function goTo(index) {
      const clip = await fetchClip(index);
      renderClip(clip);
    }

    document.getElementById("positive").addEventListener("click", () => setLabel(1));
    document.getElementById("negative").addEventListener("click", () => setLabel(0));
    document.getElementById("clear").addEventListener("click", () => setLabel(null));
    document.getElementById("prev").addEventListener("click", () => goTo(Math.max(0, currentIndex - 1)));
    document.getElementById("next").addEventListener("click", () => goTo(currentIndex + 1));

    document.addEventListener("keydown", (event) => {
      if (event.key === "1") setLabel(1);
      if (event.key === "0") setLabel(0);
      if (event.key === "u" || event.key === "U") setLabel(null);
      if (event.key === "j" || event.key === "J") goTo(currentIndex + 1);
      if (event.key === "k" || event.key === "K") goTo(Math.max(0, currentIndex - 1));
    });

    goTo(__START_INDEX__);
  </script>
</body>
</html>
"""


class ReviewHandler(BaseHTTPRequestHandler):
    store: ReviewStore
    start_index: int

    def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: str) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML.replace("__START_INDEX__", str(type(self).start_index)))
            return
        if parsed.path == "/api/clip":
            params = urllib.parse.parse_qs(parsed.query)
            index = int(params.get("index", [type(self).start_index])[0])
            index = max(0, min(index, len(type(self).store.clips) - 1))
            self._send_json(type(self).store.clip_payload(index))
            return
        if parsed.path == "/clip":
            params = urllib.parse.parse_qs(parsed.query)
            clip_id = params.get("clip_id", [None])[0]
            if clip_id is None:
                self.send_error(HTTPStatus.BAD_REQUEST, "Missing clip id.")
                return
            try:
                body = type(self).store.clip_gif_bytes(str(clip_id))
            except KeyError:
                self.send_error(HTTPStatus.NOT_FOUND, "Clip not found.")
                return
            except FileNotFoundError as exc:
                self.send_error(HTTPStatus.NOT_FOUND, str(exc))
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/gif")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/frame":
            params = urllib.parse.parse_qs(parsed.query)
            raw_path = params.get("path", [None])[0]
            if raw_path is None:
                self.send_error(HTTPStatus.BAD_REQUEST, "Missing frame path.")
                return
            frame_path = Path(urllib.parse.unquote(raw_path)).resolve()
            allowed_roots = [type(self).store.manifest_path.parent.resolve()]
            if type(self).store.data_root is not None:
                allowed_roots.append(type(self).store.data_root.resolve())
            if not any(root == frame_path or root in frame_path.parents for root in allowed_roots):
                self.send_error(HTTPStatus.FORBIDDEN, "Frame path is outside allowed roots.")
                return
            if not frame_path.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Frame not found.")
                return
            body = frame_path.read_bytes()
            mime, _ = mimetypes.guess_type(frame_path.name)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/api/label":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length) or "{}")
        index = int(payload["index"])
        index = max(0, min(index, len(type(self).store.clips) - 1))
        binary_label = payload.get("binary_label")
        if binary_label not in (None, 0, 1):
            self.send_error(HTTPStatus.BAD_REQUEST, "binary_label must be one of null, 0, or 1.")
            return
        self._send_json(type(self).store.set_label(index, binary_label))

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Review binary labels for clip-evaluation JSONL files.")
    parser.add_argument("--labels-path", required=True, help="JSONL evaluation labels file to edit in place.")
    parser.add_argument("--manifest-path", default=None, help="Manifest JSONL path with frame_paths. Inferred when omitted.")
    parser.add_argument("--data-root", default=None, help="Optional root directory prepended to relative frame_paths.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--start-index", type=int, default=None, help="Optional clip index to open first.")
    args = parser.parse_args()

    labels_path = Path(args.labels_path).resolve()
    manifest_path = (
        Path(args.manifest_path).resolve()
        if args.manifest_path
        else infer_manifest_path(labels_path).resolve()
    )
    data_root = Path(args.data_root).resolve() if args.data_root else None
    store = ReviewStore(labels_path=labels_path, manifest_path=manifest_path, data_root=data_root)

    ReviewHandler.store = store
    ReviewHandler.start_index = (
        max(0, min(args.start_index, len(store.clips) - 1))
        if args.start_index is not None
        else store.first_unlabeled_index()
    )

    server = ThreadingHTTPServer((args.host, args.port), ReviewHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Review server: {url}")
    print(f"Labels: {labels_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Data root: {data_root or store.data_root or 'manifest-relative paths'}")
    print("Open the URL in your browser. Keys: 1 positive, 0 negative, U clear, J next, K previous.")
    server.serve_forever()


if __name__ == "__main__":
    main()
