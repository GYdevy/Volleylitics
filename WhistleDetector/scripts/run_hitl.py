import tkinter as tk

from WhistleDetector.config import MATCH_NUM
from WhistleDetector.hitl.labeling import load_ambiguous
from WhistleDetector.hitl.gui import HITLLabeler


def main():
    ambig_items = load_ambiguous(MATCH_NUM)

    if not ambig_items:
        print("No ambiguous whistles 🎉")
        return

    root = tk.Tk()
    app = HITLLabeler(root, ambig_items)
    root.mainloop()


if __name__ == "__main__":
    main()
