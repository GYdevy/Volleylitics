#!/usr/bin/env python3
"""Serve docs locally and smoke-test the files GitHub Pages will publish."""

from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import sys
from threading import Thread
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import urlopen

from validate_site import DOCS_ROOT, INDEX_PATH, ValidationError, validate_site


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *_args) -> None:
        pass


def fetch(base_url: str, relative_url: str) -> bytes:
    url = urljoin(base_url, relative_url)
    try:
        with urlopen(url, timeout=5) as response:
            if response.status != 200:
                raise ValidationError(f"GET {relative_url} returned HTTP {response.status}")
            body = response.read()
    except (HTTPError, URLError, TimeoutError) as exc:
        raise ValidationError(f"GET {relative_url} failed: {exc}") from exc
    if not body:
        raise ValidationError(f"GET {relative_url} returned an empty response")
    return body


def main() -> int:
    try:
        match_count, rally_count, _warnings = validate_site()
        handler = partial(QuietHandler, directory=str(DOCS_ROOT))
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}/"
        try:
            html = fetch(base_url, "index.html").decode("utf-8")
            if '<div id="root"></div>' not in html:
                raise ValidationError("docs/index.html has no application root")
            asset_urls = re.findall(r'(?:src|href)="(\./assets/[^"]+)"', html)
            if not asset_urls:
                raise ValidationError("docs/index.html references no compiled assets")
            for asset_url in asset_urls:
                body = fetch(base_url, asset_url)
                if len(body) < 100:
                    raise ValidationError(f"Compiled asset looks truncated: {asset_url}")

            catalog = json.loads(fetch(base_url, "data/matches/index.json"))
            for group in catalog:
                for match in group["matches"]:
                    fetch(base_url, match["ralliesFile"])
                    fetch(base_url, match["detailsFile"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    except (ValidationError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"SITE SMOKE TEST FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        f"Site smoke test passed: index, {len(asset_urls)} compiled assets, "
        f"{match_count} matches, and {rally_count} rallies served over HTTP"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
