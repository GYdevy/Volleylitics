import tkinter as tk
from tkinter import ttk
import os
from pathlib import Path

from WhistleDetector.config import MATCH_NUM, BASE_OUTPUT_DIR
from WhistleDetector.audio.features import extract_features
from WhistleDetector.audio.utils import fmt_time
from WhistleDetector.hitl.labeling import append_training_row
import librosa
from WhistleDetector.config import SR

CSV_PATH = Path(BASE_OUTPUT_DIR) / "training_hitl.csv"


class HITLLabeler:
    def __init__(self, root, ambig_items):
        self.root = root
        self.ambig_items = ambig_items
        self.idx = 0

        self.root.title("Whistle Labeler")
        self.root.geometry("520x220")

        self.label = ttk.Label(root, text="", font=("Arial", 12))
        self.label.pack(pady=15)

        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="1️⃣ Whistle", command=self.accept).grid(row=0, column=0, padx=10)
        ttk.Button(btn_frame, text="2️⃣ Noise", command=self.reject).grid(row=0, column=1, padx=10)
        ttk.Button(btn_frame, text="❌ Quit", command=root.quit).grid(row=0, column=2, padx=10)

        root.bind("1", lambda e: self.accept())
        root.bind("2", lambda e: self.reject())
        root.bind("q", lambda e: root.quit())

        self.load_current()

    def load_current(self):
        if self.idx >= len(self.ambig_items):
            self.label.config(text="✅ Done!")
            return

        path, d = self.ambig_items[self.idx]
        ts = f"{fmt_time(d['start'])} → {fmt_time(d['end'])}"

        self.label.config(
            text=f"{self.idx+1}/{len(self.ambig_items)}\n{path.name}\n{ts}"
        )

        os.startfile(path)

    def _label(self, label, label_name):
        path, d = self.ambig_items[self.idx]

        # 🔹 Load audio from clip
        y, _ = librosa.load(path, sr=SR)

        _, X = extract_features(y)

        uid = (
            f"match{MATCH_NUM}_"
            f"{int(d['start'] * 1000):010d}_"
            f"{int(d['end'] * 1000):010d}"
        )

        append_training_row(uid, X, label,CSV_PATH)

        print(f"[HITL] {label_name} {uid}")

        self.idx += 1
        self.load_current()

    def accept(self):
        self._label(1, "WHISTLE")

    def reject(self):
        self._label(0, "NOISE")
