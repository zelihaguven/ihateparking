"""Which Panda3D window settings open a window on this Mac?

MetaDrive's 3-D window (``--window``) asks Panda3D for an onscreen OpenGL
window with 8x multisampling. macOS OpenGL drivers can refuse that. This
tries a plain Panda3D window with several settings, each in its own
process, and then MetaDrive itself with the first setting that worked.
Run it from your own Terminal (a window needs a desktop session):

    .venv-metadrive/bin/python scripts/metadrive_window_check.py
"""
from __future__ import annotations

import subprocess
import sys

VARIANTS = {
    "metadrive default (multisamples 8)": ["framebuffer-multisample 1", "multisamples 8"],
    "no multisampling": ["framebuffer-multisample 0", "multisamples 0"],
    "multisamples 4": ["framebuffer-multisample 1", "multisamples 4"],
    "OpenGL 3.2 core, no multisampling": ["framebuffer-multisample 0", "multisamples 0", "gl-version 3 2"],
}

PANDA = """
import sys
from panda3d.core import loadPrcFileData
for line in sys.argv[1:]:
    loadPrcFileData("", line)
loadPrcFileData("", "notify-level-display error")
from direct.showbase.ShowBase import ShowBase
base = ShowBase()
for _ in range(10):
    base.taskMgr.step()
base.destroy()
print("OPENED")
"""


def try_panda(settings: list[str]) -> tuple[bool, str]:
    run = subprocess.run([sys.executable, "-c", PANDA, *settings], capture_output=True, text=True, timeout=60)
    output = run.stdout + run.stderr
    lines = [line for line in output.splitlines() if not line.startswith("objc[")]
    return "OPENED" in run.stdout, "\n".join(lines[-4:])


def main() -> int:
    working = None
    for name, settings in VARIANTS.items():
        ok, tail = try_panda(settings)
        print(f"{'OK  ' if ok else 'FAIL'} {name}")
        if not ok:
            print("     " + tail.replace("\n", "\n     "))
        if ok and working is None:
            working = settings
    if working is None:
        print("\nNo setting opened a window: Panda3D itself cannot open one here.")
        return 1
    print(f"\nFirst working setting: {working}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
