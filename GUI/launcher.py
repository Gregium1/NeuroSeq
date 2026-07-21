from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path


def _prepare_import_path() -> None:
    """
    Make the project root importable in normal Python and PyInstaller builds.
    """
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
    else:
        root = Path(__file__).resolve().parent.parent

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> None:
    mp.freeze_support()
    _prepare_import_path()

    from GUI.neuroseq_gui import NeuroSeqGUI

    app = NeuroSeqGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
