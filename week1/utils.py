
"""Small shared helpers (JSON / text IO, console headers, seeding)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

# Make Chinese / accented output survive a Windows console with a legacy code page.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # pragma: no cover - older interpreters
    pass


def header(title: str, char: str = "=", width: int = 78) -> None:
    """Print a highlighted section header."""
    print()
    print(char * width)
    print(title)
    print(char * width)


def silence_tensorflow() -> None:
    """
    Hide TensorFlow's routine deprecation chatter so the console output stays
    readable. Call this immediately after importing tensorflow.
    """
    import logging

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        import tensorflow as tf

        tf.get_logger().setLevel(logging.ERROR)
        tf.autograph.set_verbosity(0)
    except Exception:
        pass
    try:
        from tensorflow.python.util import deprecation

        deprecation._PRINT_DEPRECATION_WARNINGS = False
    except Exception:
        pass
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


def configure_matplotlib() -> None:
    """
    Pick a plotting backend appropriate to the host process.

    From the command line we want a headless backend that only writes PNG files.
    Inside Jupyter we must leave the backend alone, otherwise plots stop rendering
    inline - so this becomes a no-op there.
    """
    import matplotlib

    if "ipykernel" in sys.modules or os.environ.get("MPLBACKEND"):
        return
    matplotlib.use("Agg")


def step(message: str) -> None:
    print(f"\n>>> {message}")


def human_bytes(num: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:,.1f} {unit}"
        num /= 1024.0
    return f"{num:,.1f} TB"


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    path = Path(path)
    ensure_dirs(path.parent)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(obj: Any) -> Any:
    # numpy scalars / arrays and tf tensors show up in metrics dicts
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def write_text(path: Path, text: str) -> None:
    path = Path(path)
    ensure_dirs(path.parent)
    path.write_text(text, encoding="utf-8")


def round_floats(mapping: dict, ndigits: int = 4) -> dict:
    out = {}
    for k, v in mapping.items():
        if isinstance(v, (int, float)):
            out[k] = round(float(v), ndigits)
        elif isinstance(v, list):
            out[k] = [round(float(x), ndigits) if isinstance(x, (int, float)) else x for x in v]
        else:
            out[k] = v
    return out


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """Render a GitHub-flavoured Markdown table."""
    rows = [[("" if c is None else str(c)) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = [
        "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([line, sep, *body])
