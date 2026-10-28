# Releasing sstsr

The published name is `sstsr`; the import name stays `tsr`.

## The version comes from the tag

`pyproject.toml` declares `dynamic = ["version"]` and hatch-vcs derives the version
from the git tag, writing `src/tsr/_version.py` at build time (git-ignored, shipped in
both artifacts so a wheel can be built from an unpacked sdist). `tsr.__version__` reads
that file, falling back to installed package metadata.

This is what makes the dry run real: **an rc tag necessarily builds an rc artifact**,
so TestPyPI only ever receives pre-releases. Before this (through 2.2.0), the version
was static in `pyproject.toml`, so tagging `v2.1.0rc1` published `2.1.0` to TestPyPI —
the rc was never exercised, and the final version number was consumed there.

Two consequences worth knowing:

- **A dirty tree or a commit after the tag builds a dev version** (`2.3.0.dev4`), not
  the tag's. That is deliberate: only a clean checkout of the tag can publish the
  release. Local segments (`+g<sha>`) are disabled, so a dry-run build is still
  uploadable to TestPyPI.
- **Two guards run before any publish job**: the built artifact's version must equal
  the tag, and an rc tag must build a pre-release (and a final tag must not).
- **In CI the tag is pinned as the version** (`SETUPTOOLS_SCM_PRETEND_VERSION`, the
  generic setuptools-scm variable — hatch-vcs does not pass the distribution name
  through, so the `..._FOR_SSTSR` form is silently ignored).
  `git describe` has to choose when several tags point at one commit, which is the
  normal flow — `vX.Y.Zrc1` and `vX.Y.Z` usually sit on the same commit — and it is not
  guaranteed to choose the one being built. It resolved to the rc in CI while resolving
  to the final tag locally, so the version is taken from the ref instead of inferred.
  The guard above then still has something to check: that the built artifact really
  carries it.

## One artifact set, verified, then published

`.github/workflows/release.yml` builds the sdist and the wheel **once**, and publishes
exactly those bytes:

1. `build` (unprivileged) builds both, checks the guards above, and writes `SHA256SUMS`
   and `manifest.json` (version, tag, commit, run, and every file's hash) beside them.
2. `verify` installs each file **alone in a clean environment** (the wheel on Python 3.10
   and 3.14 on Linux and 3.12 on macOS, the sdist on 3.12 on Linux) and runs the test
   suite against the installed package, never `src/`. Each run's report names the file,
   its hash, the interpreter, and the test outcomes.
3. An rc tag uploads the files unchanged to TestPyPI, and `tools/release_index.py`
   confirms TestPyPI serves every one with its built hash.
4. A final tag uploads them unchanged to PyPI, only after every `verify` job passed. The
   same check confirms PyPI serves them, and sstsr installed from PyPI passes the tests.
5. The GitHub release attaches the same files, `SHA256SUMS`, `manifest.json`, and
   `verification.json` (the manifest and every verify report), and the job downloads
   them back to confirm their hashes. Its notes are this version's changelog section.

Every job after `build` re-checks the files against `SHA256SUMS`, and `SHA256SUMS`
against the digest `build` recorded, before it acts. Immediately before each upload and
the release, the job resolves the tag again and stops unless it still points to the
run's commit. Only the two publishing jobs hold `id-token: write`, only the release job
holds `contents: write`, and every third-party action is pinned to a full commit SHA.

Anyone can check an installed release against the published hashes:

```bash
gh release download vX.Y.Z --repo personalrobotics/sstsr --pattern SHA256SUMS
pip download --no-deps "sstsr==X.Y.Z" -d dl && pip download --no-deps --no-binary :all: "sstsr==X.Y.Z" -d dl
(cd dl && shasum -a 256 -c ../SHA256SUMS)
```

## Steps

1. Land everything for the release. Move the changelog's `[Unreleased]` heading to
   `## [X.Y.Z] — YYYY-MM`, summarising highlights and anything now withheld.
   Do **not** edit a version number anywhere; there is none to edit.
2. Check the gate locally:
   ```bash
   uv run pytest tests/ -q
   uv run ruff check . && uv run ruff format --check .
   TSR_MATRIX_SCALE=10 uv run pytest tests/tsr/hands/test_grasp_matrix.py \
       tests/tsr/hands/test_grasp_mutation_gate.py
   ```
3. Tag the release candidate and push it. Once every verify job passes, this
   publishes to **TestPyPI**:
   ```bash
   git tag -a vX.Y.Zrc1 -m "sstsr X.Y.Zrc1" && git push origin vX.Y.Zrc1
   ```
4. Verify from TestPyPI in a clean venv (its own index first, PyPI only for
   dependencies):
   ```bash
   uv venv /tmp/rc && VIRTUAL_ENV=/tmp/rc uv pip install \
     --index-url https://test.pypi.org/simple/ \
     --extra-index-url https://pypi.org/simple/ \
     --index-strategy unsafe-best-match "sstsr==X.Y.Zrc1"
   /tmp/rc/bin/python -c "import tsr; print(tsr.__version__)"
   ```
   Exercise what the release actually changed, not just the import.
5. Tag the final release on the rc's commit. Once every verify job passes, this
   publishes to **PyPI** and creates the GitHub Release with the files and their hashes:
   ```bash
   git tag -a vX.Y.Z -m "sstsr X.Y.Z" && git push origin vX.Y.Z
   ```

Publishing uses Trusted Publishing (OIDC): there are no tokens, and the publish jobs
run in the `testpypi-release` / `release` GitHub environments.

## If an rc needs a fix

Land the fix and tag `rc2`. Never move or delete a tag while a run for it exists, and
never move a published one: the artifact for a given version can never be replaced on
PyPI or TestPyPI.
