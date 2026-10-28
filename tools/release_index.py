"""Confirm that a package index serves exactly the files a release built (release.yml).

Reads the index's JSON API for ``sstsr`` at ``--version``, waiting while a new upload is
not yet listed, and requires its files to be exactly those in ``--sums`` (SHA256SUMS),
each with the recorded hash: the hash the index reports, and the hash of the bytes it
serves when downloaded. Standard library only.

    python3 -I tools/release_index.py https://test.pypi.org --version 4.1.0rc1 \
        --sums release/SHA256SUMS
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request

WAIT_S = 600
"""How long a new upload may take to be listed before the check fails."""
POLL_S = 30


def _listed(index: str, version: str) -> list[dict]:
    url = f"{index}/pypi/sstsr/{version}/json"
    deadline = time.monotonic() + WAIT_S
    while True:
        try:
            with urllib.request.urlopen(url) as response:
                return json.load(response)["urls"]
        except urllib.error.HTTPError as error:
            if error.code != 404 or time.monotonic() > deadline:
                raise SystemExit(f"{url}: {error}") from error
        time.sleep(POLL_S)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("index", help="the index's root, e.g. https://pypi.org")
    parser.add_argument("--version", required=True)
    parser.add_argument("--sums", required=True, help="the release's SHA256SUMS")
    args = parser.parse_args()
    with open(args.sums) as sums:
        built = {name: digest for digest, name in (line.split() for line in sums)}
    served = {entry["filename"]: entry for entry in _listed(args.index, args.version)}
    if set(served) != set(built):
        raise SystemExit(f"{args.index} serves {sorted(served)}, the release built {sorted(built)}")
    for name, entry in sorted(served.items()):
        if entry["digests"]["sha256"] != built[name]:
            raise SystemExit(f"{name}: {args.index} reports another hash")
        with urllib.request.urlopen(entry["url"]) as response:
            downloaded = hashlib.sha256(response.read()).hexdigest()
        if downloaded != built[name]:
            raise SystemExit(f"{name}: {args.index} serves other bytes")
        print(f"{name}: {downloaded}, as built")
    return 0


if __name__ == "__main__":
    sys.exit(main())
