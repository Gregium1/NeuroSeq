"""Provide a GUI entry point for launching the NeuroSeq desktop application.

Author
------
Greg Owanga
Gregium Theory
"""

# GUI/main.py
from __future__ import annotations
import multiprocessing as mp
from .neuroseq_gui import NeuroSeqGUI

def main():
    mp.freeze_support()
    app=NeuroSeqGUI(); app.mainloop()

if __name__ == "__main__": main()
