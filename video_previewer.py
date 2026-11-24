"""Lightweight video preview and set-range picker for the Windows GUI."""

from __future__ import annotations

import argparse
import json
import math
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk


COLORS = {
    "background": "#0b1020",
    "panel": "#121a2b",
    "input": "#172137",
    "text": "#f3f6fb",
    "muted": "#94a3b8",
    "green": "#22c55e",
    "blue": "#38bdf8",
    "red": "#ef4444",
}


def parse_timestamp(value: str) -> float:
    parts = value.strip().split(":")
    if not parts or len(parts) > 3:
        raise ValueError("Use seconds, MM:SS, or HH:MM:SS.")
    try:
        numbers = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError("Use seconds, MM:SS, or HH:MM:SS.") from exc
    if any(number < 0 for number in numbers):
        raise ValueError("The timestamp cannot be negative.")
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class VideoPreviewer(tk.Tk):
    def __init__(
        self,
        video_path: Path,
        output_path: Path,
        initial_ranges: list[list[float]],
    ) -> None:
        super().__init__()
        self.video_path = video_path
        self.output_path = output_path
        self.capture = cv2.VideoCapture(str(video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.fps = 30.0
        frame_count = float(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = frame_count / self.fps if frame_count > 0 else 0.0
        if self.duration <= 0:
            raise RuntimeError("Could not determine the video duration.")

        self.current_time = 0.0
        self.pending_start: float | None = None
        self.ranges = [
            [max(0.0, float(start)), min(self.duration, float(end))]
            for start, end in initial_ranges
            if float(end) > float(start)
        ]
        self.playing = False
        self.play_anchor_wall = 0.0
        self.play_anchor_time = 0.0
        self.slider_job: str | None = None
        self.updating_slider = False
        self.photo: ImageTk.PhotoImage | None = None

        self.title(f"Set Range Preview — {video_path.name}")
        self.geometry("1040x820")
        self.minsize(850, 690)
        self.configure(bg=COLORS["background"])
        self.protocol("WM_DELETE_WINDOW", self.close)
        self._build_ui()
        self._seek(0.0)
        self.after(30, self._tick)

    def _button(self, parent: tk.Misc, text: str, command, accent: bool = False):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=COLORS["green"] if accent else COLORS["input"],
            fg="#07140b" if accent else COLORS["text"],
            activebackground=COLORS["blue"],
            activeforeground="#07111c",
            relief="flat",
            padx=12,
            pady=7,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
        )

    def _build_ui(self) -> None:
        header = tk.Frame(self, bg=COLORS["background"])
        header.pack(fill="x", padx=20, pady=(14, 8))
        tk.Label(
            header,
            text="SET RANGE PREVIEW",
            bg=COLORS["background"],
            fg=COLORS["green"],
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header,
            text="Find each set boundary, then mark its start and end",
            bg=COLORS["background"],
            fg=COLORS["text"],
            font=("Segoe UI", 18, "bold"),
        ).pack(anchor="w")

        self.video_label = tk.Label(
            self,
            bg="#05070d",
            fg=COLORS["muted"],
            text="Loading video frame…",
        )
        self.video_label.pack(fill="both", expand=True, padx=20)

        controls = tk.Frame(self, bg=COLORS["panel"], padx=14, pady=10)
        controls.pack(fill="x", padx=20, pady=(10, 8))

        self.time_var = tk.StringVar(value="00:00:00.000")
        self.time_label = tk.Label(
            controls,
            textvariable=self.time_var,
            bg=COLORS["panel"],
            fg=COLORS["text"],
            font=("Consolas", 13, "bold"),
        )
        self.time_label.pack(anchor="w")

        self.slider_var = tk.DoubleVar(value=0.0)
        self.slider = ttk.Scale(
            controls,
            from_=0.0,
            to=self.duration,
            variable=self.slider_var,
            command=self._slider_changed,
        )
        self.slider.pack(fill="x", pady=(4, 8))

        row = tk.Frame(controls, bg=COLORS["panel"])
        row.pack(fill="x")
        self.play_button = self._button(row, "Play (silent)", self.toggle_play)
        self.play_button.pack(side="left", padx=(0, 5))
        for label, delta in (("−10s", -10), ("−1s", -1), ("+1s", 1), ("+10s", 10)):
            self._button(row, label, lambda amount=delta: self.skip(amount)).pack(
                side="left", padx=3
            )

        self.goto_var = tk.StringVar(value="00:00:00")
        goto = tk.Entry(
            row,
            textvariable=self.goto_var,
            width=14,
            bg=COLORS["input"],
            fg=COLORS["text"],
            insertbackground=COLORS["text"],
            relief="flat",
            font=("Consolas", 10),
        )
        goto.pack(side="left", padx=(16, 4), ipady=7)
        goto.bind("<Return>", lambda _event: self.go_to_timestamp())
        self._button(row, "Go", self.go_to_timestamp).pack(side="left")
        tk.Label(
            row,
            text=f"of {format_timestamp(self.duration)}",
            bg=COLORS["panel"],
            fg=COLORS["muted"],
        ).pack(side="right")

        ranges_panel = tk.Frame(self, bg=COLORS["panel"], padx=14, pady=10)
        ranges_panel.pack(fill="x", padx=20, pady=(0, 16))
        mark_row = tk.Frame(ranges_panel, bg=COLORS["panel"])
        mark_row.pack(fill="x", pady=(0, 8))
        self._button(mark_row, "Set start", self.set_start).pack(side="left")
        self._button(mark_row, "Set end", self.set_end).pack(side="left", padx=6)
        self.start_var = tk.StringVar(value="Start not marked")
        tk.Label(
            mark_row,
            textvariable=self.start_var,
            bg=COLORS["panel"],
            fg=COLORS["blue"],
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=8)

        self._button(mark_row, "Save & use", self.save, accent=True).pack(side="right")
        self._button(mark_row, "Clear", self.clear_ranges).pack(side="right", padx=6)
        self._button(mark_row, "Delete selected", self.delete_selected).pack(side="right")

        self.range_list = tk.Listbox(
            ranges_panel,
            height=4,
            bg=COLORS["input"],
            fg=COLORS["text"],
            selectbackground=COLORS["blue"],
            selectforeground="#07111c",
            relief="flat",
            font=("Consolas", 10),
        )
        self.range_list.pack(fill="x")
        self._refresh_ranges()

    def _display_frame(self, frame) -> None:
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width = max(self.video_label.winfo_width(), 640)
        height = max(self.video_label.winfo_height(), 360)
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.video_label.configure(image=self.photo, text="")

    def _set_current_time(self, seconds: float) -> None:
        self.current_time = max(0.0, min(float(seconds), self.duration))
        self.time_var.set(format_timestamp(self.current_time))
        self.updating_slider = True
        self.slider_var.set(self.current_time)
        self.updating_slider = False

    def _seek(self, seconds: float) -> None:
        self.pause()
        target = max(0.0, min(float(seconds), self.duration))
        self.capture.set(cv2.CAP_PROP_POS_MSEC, target * 1000.0)
        ok, frame = self.capture.read()
        if ok:
            self._display_frame(frame)
        self._set_current_time(target)

    def _slider_changed(self, value: str) -> None:
        if self.updating_slider:
            return
        if self.slider_job is not None:
            self.after_cancel(self.slider_job)
        self.slider_job = self.after(80, lambda: self._seek(float(value)))

    def toggle_play(self) -> None:
        if self.playing:
            self.pause()
            return
        if self.current_time >= self.duration - 0.05:
            self._seek(0.0)
        self.playing = True
        self.play_anchor_wall = time.monotonic()
        self.play_anchor_time = self.current_time
        self.play_button.configure(text="Pause")

    def pause(self) -> None:
        self.playing = False
        if hasattr(self, "play_button"):
            self.play_button.configure(text="Play (silent)")

    def _tick(self) -> None:
        if self.playing:
            target = self.play_anchor_time + (time.monotonic() - self.play_anchor_wall)
            if target >= self.duration:
                self._seek(self.duration)
            else:
                frame = None
                while self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0 < target:
                    if not self.capture.grab():
                        self.pause()
                        break
                    ok, candidate = self.capture.retrieve()
                    if ok:
                        frame = candidate
                if frame is not None:
                    self._display_frame(frame)
                self._set_current_time(target)
        self.after(30, self._tick)

    def skip(self, seconds: float) -> None:
        self._seek(self.current_time + seconds)

    def go_to_timestamp(self) -> None:
        try:
            self._seek(parse_timestamp(self.goto_var.get()))
        except ValueError as exc:
            messagebox.showerror("Invalid timestamp", str(exc), parent=self)

    def set_start(self) -> None:
        self.pending_start = self.current_time
        self.start_var.set(f"Start: {format_timestamp(self.pending_start)}")

    def set_end(self) -> None:
        if self.pending_start is None:
            messagebox.showinfo("Set start first", "Mark the set start first.", parent=self)
            return
        if self.current_time <= self.pending_start:
            messagebox.showerror(
                "Invalid set range",
                "The set end must be after its start.",
                parent=self,
            )
            return
        self.ranges.append([self.pending_start, self.current_time])
        self.ranges.sort(key=lambda item: item[0])
        self.pending_start = None
        self.start_var.set("Start not marked")
        self._refresh_ranges()

    def _refresh_ranges(self) -> None:
        self.range_list.delete(0, tk.END)
        for index, (start, end) in enumerate(self.ranges, 1):
            self.range_list.insert(
                tk.END,
                f"Set {index}:  {format_timestamp(start)} - {format_timestamp(end)}"
                f"   ({format_timestamp(end - start)})",
            )

    def delete_selected(self) -> None:
        selection = self.range_list.curselection()
        if not selection:
            return
        del self.ranges[selection[0]]
        self._refresh_ranges()

    def clear_ranges(self) -> None:
        self.ranges.clear()
        self.pending_start = None
        self.start_var.set("Start not marked")
        self._refresh_ranges()

    def save(self) -> None:
        if self.pending_start is not None:
            messagebox.showinfo(
                "Incomplete range",
                "Mark the current set end or clear its start before saving.",
                parent=self,
            )
            return
        if not self.ranges:
            answer = messagebox.askyesno(
                "Use full video?",
                "No ranges are marked. Clear the set ranges and use the full video?",
                parent=self,
            )
            if not answer:
                return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        text = "\n".join(
            f"{format_timestamp(start)} - {format_timestamp(end)}"
            for start, end in self.ranges
        )
        self.output_path.write_text(text, encoding="utf-8")
        self.close()

    def close(self) -> None:
        self.pause()
        self.capture.release()
        self.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview a video and mark set ranges")
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--initial-ranges-json", default="[]")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        assert parse_timestamp("01:02:03.5") == 3723.5
        assert format_timestamp(3723.5) == "01:02:03.500"
        return 0

    try:
        initial_ranges = json.loads(args.initial_ranges_json)
        if not isinstance(initial_ranges, list):
            raise ValueError("Initial ranges must be a list")
        app = VideoPreviewer(args.video.resolve(), args.output.resolve(), initial_ranges)
        app.mainloop()
        return 0
    except Exception as exc:
        try:
            messagebox.showerror("Previewer error", str(exc))
        except Exception:
            print(f"Previewer error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
