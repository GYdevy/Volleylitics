"""Model chooser with saved Volleylitics benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parent
COLORS = {
    "background": "#0b1020",
    "panel": "#121a2b",
    "input": "#172137",
    "text": "#f3f6fb",
    "muted": "#94a3b8",
    "green": "#22c55e",
    "blue": "#38bdf8",
}


@dataclass(frozen=True)
class ModelOption:
    path: Path
    label: str
    details: str


def ball_benchmarks() -> dict[str, dict[str, float | str]]:
    metrics: dict[str, dict[str, float | str]] = {}
    ranking_path = ROOT / "model_evaluation" / "model_ranking.csv"
    speed_path = ROOT / "model_evaluation" / "speed_benchmark.csv"
    if ranking_path.is_file():
        with ranking_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "success":
                    continue
                metrics.setdefault(row["model"].replace("\\", "/"), {}).update(
                    f1=float(row["f1"]),
                    precision=float(row["precision"]),
                    recall=float(row["recall"]),
                    map50=float(row["map50"]),
                )
    if speed_path.is_file():
        with speed_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") != "success":
                    continue
                metrics.setdefault(row["model"].replace("\\", "/"), {}).update(
                    fps=float(row["fps_from_median"]),
                    latency=float(row["median_ms"]),
                    memory=float(row["peak_gpu_memory_mb"]),
                    sha=row["sha256"],
                )
    return metrics


def local_path_for_benchmark(stored_path: str) -> Path | None:
    normalized = stored_path.replace("\\", "/")
    marker = "/Volleylitics/"
    if marker in normalized:
        relative = normalized.split(marker, 1)[1]
        candidate = ROOT / Path(relative)
        if candidate.is_file():
            return candidate.resolve()
    candidate = ROOT / "model" / Path(normalized).name
    return candidate.resolve() if candidate.is_file() else None


def make_ball_options() -> list[ModelOption]:
    options: list[ModelOption] = []
    seen_hashes: set[str] = set()
    default_model = (ROOT / "models" / "temp_best_ball.pt").resolve()
    for stored_path, metric in ball_benchmarks().items():
        path = local_path_for_benchmark(stored_path)
        if path is None:
            continue
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        sha = hasher.hexdigest()
        if sha in seen_hashes:
            continue
        seen_hashes.add(sha)
        if path.name == "temp_best_ball.pt" and default_model.is_file():
            path = default_model
        f1 = metric.get("f1")
        fps = metric.get("fps")
        summary = []
        if f1 is not None:
            summary.append(f"F1 {f1 * 100:.2f}%")
        if fps is not None:
            summary.append(f"{fps:.2f} FPS")
        suffix = " — " + ", ".join(summary) if summary else ""
        details = (
            f"Precision {metric.get('precision', 0) * 100:.2f}%   "
            f"Recall {metric.get('recall', 0) * 100:.2f}%   "
            f"F1 {metric.get('f1', 0) * 100:.2f}%   "
            f"mAP50 {metric.get('map50', 0) * 100:.2f}%\n"
            f"Median {metric.get('latency', 0):.2f} ms   "
            f"{metric.get('fps', 0):.2f} FPS   "
            f"Peak GPU memory {metric.get('memory', 0):.1f} MB\n"
            "Saved 960px benchmark; actual full-video speed also depends on decoding and tracking."
        )
        options.append(ModelOption(path, f"{path.name}{suffix}", details))
    options.sort(key=lambda item: item.path != default_model)
    return options


def known_options() -> dict[str, list[ModelOption]]:
    whistle = [
        ModelOption(
            (ROOT / "models" / "best_resnet_whistle_mix.pth").resolve(),
            "best_resnet_whistle_mix.pth — F1 98.87%",
            "Validation: precision 99.06%, recall 98.69%, F1 98.87%, accuracy 99.25%.",
        ),
        ModelOption(
            (ROOT / "rally_segmentator" / "models_dir" / "best_resnet_f1.pth").resolve(),
            "best_resnet_f1.pth — F1 99.06%",
            "Validation: precision 99.25%, recall 98.88%, F1 99.06%, accuracy 99.61%.\n"
            "This checkpoint recorded a validation threshold of 0.35; the current runtime threshold remains unchanged.",
        ),
    ]
    rally = [
        ModelOption(
            (ROOT / "models" / "rally_model_best.pth").resolve(),
            "rally_model_best.pth — current trained classifier",
            "No comparable rally-model benchmark was saved in model_evaluation.",
        )
    ]
    return {
        "whistle": [option for option in whistle if option.path.is_file()],
        "rally": [option for option in rally if option.path.is_file()],
        "ball": make_ball_options(),
    }


class ModelSelector(tk.Tk):
    def __init__(self, output: Path, initial: dict[str, str]) -> None:
        super().__init__()
        self.output = output
        self.options = known_options()
        self.selected_paths: dict[str, Path] = {}
        self.combos: dict[str, ttk.Combobox] = {}
        self.lookup: dict[str, dict[str, ModelOption]] = {}
        self.detail_vars: dict[str, tk.StringVar] = {}

        self.title("Choose Volleylitics models")
        self.geometry("820x600")
        self.minsize(720, 540)
        self.configure(bg=COLORS["background"])
        self._build(initial)

    def _build(self, initial: dict[str, str]) -> None:
        tk.Label(
            self, text="MODEL SELECTION", bg=COLORS["background"], fg=COLORS["green"],
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", padx=22, pady=(16, 2))
        tk.Label(
            self, text="Choose the models used for this processing run",
            bg=COLORS["background"], fg=COLORS["text"], font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w", padx=22, pady=(0, 5))
        tk.Label(
            self,
            text="Benchmark figures are measurements saved in model_evaluation, not estimates.",
            bg=COLORS["background"], fg=COLORS["muted"], font=("Segoe UI", 9),
        ).pack(anchor="w", padx=22, pady=(0, 10))

        panel = tk.Frame(self, bg=COLORS["panel"], padx=14, pady=10)
        panel.pack(fill="both", expand=True, padx=20, pady=(0, 16))
        panel.columnconfigure(0, weight=1)
        for row, (key, title, pattern) in enumerate(
            (
                ("whistle", "WHISTLE CNN", "PyTorch checkpoints|*.pth|All files|*.*"),
                ("rally", "RALLY CLASSIFIER", "PyTorch checkpoints|*.pth|All files|*.*"),
                ("ball", "BALL DETECTOR", "YOLO checkpoints|*.pt|All files|*.*"),
            )
        ):
            base = row * 3
            tk.Label(
                panel, text=title, bg=COLORS["panel"], fg=COLORS["muted"],
                font=("Segoe UI", 8, "bold"),
            ).grid(row=base, column=0, sticky="w", pady=(7, 2))
            initial_path = Path(initial.get(key, "")).resolve() if initial.get(key) else None
            if initial_path and initial_path.is_file() and not any(
                option.path == initial_path for option in self.options[key]
            ):
                self.options[key].append(
                    ModelOption(
                        initial_path,
                        f"Custom — {initial_path.name}",
                        f"Custom model: {initial_path}\nNo saved benchmark is linked to this file.",
                    )
                )
            option_lookup = {option.label: option for option in self.options[key]}
            self.lookup[key] = option_lookup
            combo = ttk.Combobox(panel, values=list(option_lookup), state="readonly")
            combo.grid(row=base + 1, column=0, sticky="ew", padx=(0, 94), ipady=5)
            self.combos[key] = combo
            browse = tk.Button(
                panel, text="Browse…", command=lambda k=key, p=pattern: self.browse(k, p),
                bg=COLORS["input"], fg=COLORS["text"], relief="flat",
                font=("Segoe UI", 9, "bold"), cursor="hand2",
            )
            browse.grid(row=base + 1, column=0, sticky="e", ipadx=9, ipady=4)
            detail_var = tk.StringVar()
            self.detail_vars[key] = detail_var
            tk.Label(
                panel, textvariable=detail_var, justify="left", anchor="w",
                bg=COLORS["panel"], fg=COLORS["muted"], font=("Segoe UI", 8),
            ).grid(row=base + 2, column=0, sticky="ew", pady=(3, 3))
            combo.bind("<<ComboboxSelected>>", lambda _event, k=key: self.changed(k))

            selected = next(
                (option for option in self.options[key] if initial_path == option.path),
                self.options[key][0] if self.options[key] else None,
            )
            if selected:
                combo.set(selected.label)
                self.selected_paths[key] = selected.path
                detail_var.set(selected.details)

        save = tk.Button(
            panel, text="Use these models", command=self.save,
            bg=COLORS["green"], fg="#07140b", activebackground=COLORS["blue"],
            relief="flat", font=("Segoe UI", 10, "bold"), cursor="hand2",
        )
        save.grid(row=10, column=0, sticky="ew", pady=(12, 4), ipady=8)

    def changed(self, key: str) -> None:
        option = self.lookup[key].get(self.combos[key].get())
        if option:
            self.selected_paths[key] = option.path
            self.detail_vars[key].set(option.details)

    def browse(self, key: str, pattern: str) -> None:
        parts = pattern.split("|")
        filetypes = [(parts[index], parts[index + 1]) for index in range(0, len(parts), 2)]
        selected = filedialog.askopenfilename(
            title=f"Choose {key} model", initialdir=str(ROOT), filetypes=filetypes,
        )
        if not selected:
            return
        path = Path(selected).resolve()
        label = f"Custom — {path.name}"
        option = ModelOption(path, label, f"Custom model: {path}\nNo saved benchmark is linked to this file.")
        self.lookup[key][label] = option
        self.combos[key].configure(values=list(self.lookup[key]))
        self.combos[key].set(label)
        self.changed(key)

    def save(self) -> None:
        missing = [key for key in ("whistle", "rally", "ball") if key not in self.selected_paths]
        if missing:
            messagebox.showerror("Missing model", f"Choose: {', '.join(missing)}", parent=self)
            return
        for key, path in self.selected_paths.items():
            if not path.is_file():
                messagebox.showerror("Missing model", f"{key.title()} model not found:\n{path}", parent=self)
                return
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(
            json.dumps({key: str(path) for key, path in self.selected_paths.items()}, indent=2),
            encoding="utf-8",
        )
        self.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Choose Volleylitics inference models")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-json", default="{}")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        options = known_options()
        assert options["whistle"] and options["rally"] and options["ball"]
        assert any("F1" in option.label and "FPS" in option.label for option in options["ball"])
        return 0
    try:
        initial = json.loads(args.initial_json)
        app = ModelSelector(args.output.resolve(), initial)
        app.mainloop()
        return 0
    except Exception as exc:
        try:
            messagebox.showerror("Model selector error", str(exc))
        except Exception:
            print(f"Model selector error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
