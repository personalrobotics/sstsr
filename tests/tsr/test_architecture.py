# SPDX-License-Identifier: BSD-2-Clause
# Authors: Siddhartha Srinivasa and contributors to TSR

"""Architecture-enforcement tests (see docs/ARCHITECTURE.md).

These freeze the layering so a careless future import can't rot it:

* Layer 0 (``tsr.core``) is the pure-math heart and must not import from any
  upper layer (template, hands, placement, io, sampling, viser).
* ``import tsr`` must stay lightweight — it must not transitively pull the
  optional visualization dependency (Viser), and importing sstsr must never start a
  server (#77). PyVista was removed in 3.0 (#140), so its absence is checked too: a
  reintroduced import would quietly restore the dependency the removal deleted.
"""

import ast
import subprocess
import sys
from pathlib import Path

import tsr.core

_UPPER_LAYERS = {"template", "hands", "placement", "io", "sampling", "viser"}


def _imported_names(py_file: Path):
    """Yield top-level module tokens imported by a source file."""
    tree = ast.parse(py_file.read_text(), filename=str(py_file))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            # 'from tsr.template import X' -> module 'tsr.template'
            # 'from ..hands import Y'      -> level>0, module 'hands'
            yield ("." * node.level) + (node.module or "")


def test_core_has_no_upward_imports():
    core_dir = Path(tsr.core.__file__).parent
    offenders = []
    for py_file in core_dir.rglob("*.py"):
        for name in _imported_names(py_file):
            tail = name.lstrip(".")
            head = tail.split(".")[-1] if name.startswith(".") else tail
            # Catch both absolute 'tsr.hands...' and relative '..hands' forms.
            if any(part in _UPPER_LAYERS for part in (head, tail.split(".")[0])):
                if tail.startswith("tsr.") or name.startswith("."):
                    offenders.append(f"{py_file.name}: imports {name}")
    assert not offenders, "tsr.core must not import upper layers:\n" + "\n".join(offenders)


def test_import_tsr_does_not_pull_visualization_deps():
    code = "import tsr, sys; print([m for m in ('viser', 'pyvista', 'vtk', 'matplotlib') if m in sys.modules])"
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "[]", f"`import tsr` unexpectedly imported visualization deps: {out}"


def test_no_module_imports_the_removed_pyvista_backend():
    # The 3.0 removal (#140) is only real if nothing quietly imports it back.
    src = Path(tsr.__file__).parent
    offenders = [
        f"{py.relative_to(src)}: imports {name}"
        for py in src.rglob("*.py")
        for name in _imported_names(py)
        if name.split(".")[0] in {"pyvista", "vtk"} or name.endswith("tsr.viz") or name == "viz"
    ]
    assert not offenders, "the PyVista backend was removed in 3.0:\n" + "\n".join(offenders)


def test_viser_backend_is_not_reachable_from_the_public_api():
    # tsr.viser must be imported explicitly; nothing in the package pulls it in, so
    # `import tsr` cannot start a server or require the optional extra (#77).
    code = "import tsr, sys; print('tsr.viser' in sys.modules)"
    assert subprocess.check_output([sys.executable, "-c", code], text=True).strip() == "False"


def _pyproject_table(header: str) -> str:
    """One pyproject table as raw text (``tomllib`` is stdlib only from 3.11). It is
    found from this file, not from ``tsr``, which may be an installed artifact's."""
    text = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    rest = text[text.index(f"[{header}]") + len(header) + 2 :]
    return rest[: rest.index("\n[")]


def test_both_visualization_extras_install_viser_and_not_pyvista():
    # `viz` is kept as a name so an existing `sstsr[viz]` install keeps working, but in
    # 3.0 it means Viser (#140). If either extra ever names PyVista again, the removal
    # has been undone at the packaging layer even if the code stays clean.
    extras = _pyproject_table("project.optional-dependencies")
    for extra in ("viz", "viser"):
        block = extras[extras.index(f"{extra} = [") : extras.index("]", extras.index(f"{extra} = ["))]
        assert "viser" in block, f"the {extra} extra must install Viser"
        assert "pyvista" not in block and "matplotlib" not in block, f"the {extra} extra still names PyVista"


def test_core_dependencies_stay_visualization_free():
    core = _pyproject_table("project")
    core = core[core.index("dependencies = [") : core.index("]", core.index("dependencies = ["))]
    for package in ("viser", "pyvista", "matplotlib", "Pillow"):
        assert package not in core, f"{package} must not be a core dependency"
