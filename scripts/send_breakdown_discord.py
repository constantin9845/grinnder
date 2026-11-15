#!/usr/bin/env python3
"""Send a time-breakdown JSON (TimeBreakdown.save) to a Discord webhook.

Posts a short summary message with two attachments:
  - the JSON file itself
  - breakdown.txt: the full median table (same as print_summary), if
    grinnder is importable

The webhook URL comes from --webhook or the DISCORD_WEBHOOK_URL environment
variable (keep it out of the repo and out of shell history if you can).

Usage, after a training run:
    python examples/train_igb_metis.py ... --breakdown_json run.json
    python scripts/send_breakdown_discord.py run.json --message "grouped, 16 parts"

    # check what would be sent, without sending:
    python scripts/send_breakdown_discord.py run.json --dry-run
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import socket
import statistics
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

# Discord's attachment limit without server boosts; larger files are rejected.
MAX_ATTACHMENT_BYTES = 10 * 1024 * 1024
MAX_CONTENT_CHARS = 2000
TOP = ("ssd", "ftl", "transfer", "compute", "other")


def summary_line(data: dict) -> str:
    """Median wall time and top-level shares, skipping the warm-up epoch."""
    epochs = data.get("epochs") or []
    if not epochs:
        return "no epochs recorded"
    use = epochs[1:] if len(epochs) > 1 else epochs
    wall = statistics.median(e["wall_s"] for e in use)
    parts = []
    for c in TOP:
        vals = [e.get("totals", {}).get(c, 0.0) for e in use]
        v = statistics.median(vals)
        if v > 0:
            parts.append(f"{c} {v:.2f}s ({100 * v / wall:.0f}%)" if wall else f"{c} {v:.2f}s")
    n = len(use)
    return (f"median of {n} epoch{'s' if n > 1 else ''}: {wall:.2f}s/epoch | "
            + " | ".join(parts))


def render_table(data: dict) -> Optional[str]:
    """Full breakdown table as text, via grinnder's own printer."""
    try:
        repo = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo))
        from grinnder.stats.breakdown import TimeBreakdown
    except Exception as e:  # grinnder not importable here: send the JSON only
        print(f"note: no table attachment ({e})", file=sys.stderr)
        return None
    bd = TimeBreakdown()
    bd.epochs = data.get("epochs") or []
    if not bd.epochs:
        return None
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        bd.print_summary()
        print("\n\n# Per epoch")
        for i, rec in enumerate(bd.epochs):
            bd.print_epoch(rec, title=f"epoch {i}")
    return buf.getvalue()


def multipart(payload: dict, files: List[Tuple[str, bytes, str]]) -> Tuple[bytes, str]:
    boundary = f"----grinnder{uuid.uuid4().hex}"
    out = io.BytesIO()

    def part(headers: str, body: bytes) -> None:
        out.write(f"--{boundary}\r\n{headers}\r\n\r\n".encode())
        out.write(body)
        out.write(b"\r\n")

    part('Content-Disposition: form-data; name="payload_json"\r\n'
         "Content-Type: application/json",
         json.dumps(payload).encode())
    for i, (name, body, ctype) in enumerate(files):
        part(f'Content-Disposition: form-data; name="files[{i}]"; filename="{name}"\r\n'
             f"Content-Type: {ctype}", body)
    out.write(f"--{boundary}--\r\n".encode())
    return out.getvalue(), f"multipart/form-data; boundary={boundary}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("json_path", type=Path, help="breakdown JSON written by TimeBreakdown.save")
    ap.add_argument("--webhook", default=os.environ.get("DISCORD_WEBHOOK_URL"),
                    help="Discord webhook URL (default: $DISCORD_WEBHOOK_URL)")
    ap.add_argument("--message", default="", help="extra text, e.g. the run's config")
    ap.add_argument("--no-table", action="store_true", help="attach the JSON only")
    ap.add_argument("--no-host", action="store_true", help="leave the hostname out")
    ap.add_argument("--dry-run", action="store_true", help="print what would be sent")
    args = ap.parse_args()

    if not args.json_path.is_file():
        print(f"error: {args.json_path} not found", file=sys.stderr)
        return 1
    raw = args.json_path.read_bytes()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"error: {args.json_path} is not valid JSON ({e})", file=sys.stderr)
        return 1

    header = f"**Time breakdown** `{args.json_path.name}`"
    if not args.no_host:
        header += f" on `{socket.gethostname()}`"
    lines = [header]
    meta = data.get("meta") or {}
    if meta:
        lines.append(" · ".join(f"{k}={v}" for k, v in meta.items()))
    if args.message:
        lines.append(args.message)
    lines.append(f"```{summary_line(data)}```")
    content = "\n".join(lines)[:MAX_CONTENT_CHARS]

    files = [(args.json_path.name, raw, "application/json")]
    if not args.no_table:
        table = render_table(data)
        if table:
            files.append(("breakdown.txt", table.encode(), "text/plain"))
    for name, body, _ in files:
        if len(body) > MAX_ATTACHMENT_BYTES:
            print(f"error: {name} is {len(body) / 2**20:.1f} MiB, over Discord's "
                  f"{MAX_ATTACHMENT_BYTES / 2**20:.0f} MiB limit", file=sys.stderr)
            return 1

    payload = {"content": content, "allowed_mentions": {"parse": []}}

    if args.dry_run:
        print(content)
        for name, body, _ in files:
            print(f"[attachment] {name} ({len(body):,} bytes)")
        return 0

    if not args.webhook:
        print("error: no webhook (pass --webhook or set DISCORD_WEBHOOK_URL)", file=sys.stderr)
        return 1

    body, ctype = multipart(payload, files)
    req = urllib.request.Request(
        args.webhook,
        data=body,
        method="POST",
        # Discord (Cloudflare) rejects urllib's default User-Agent.
        headers={"Content-Type": ctype, "User-Agent": "breakdown-notifier/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f"sent ({resp.status})")
    except urllib.error.HTTPError as e:
        print(f"error: Discord returned {e.code}: {e.read().decode(errors='replace')[:500]}",
              file=sys.stderr)
        return 1
    except urllib.error.URLError as e:
        print(f"error: could not reach Discord ({e.reason})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
