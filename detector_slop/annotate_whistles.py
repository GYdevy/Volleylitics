import sys
import json
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import QUrl, Qt


VIDEO_PATH = r"/videos/match3.mp4"
OUT_JSON   = "whistles_match3.json"
MATCH_ID   = "match3"


class WhistleAnnotator(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Whistle Annotator")
        self.resize(1600, 900)
        self._initial_seek_done = False
        # ===============================
        # LOAD EXISTING DATA (Resume Mode)
        # ===============================
        import os

        if os.path.exists(OUT_JSON):
            with open(OUT_JSON, "r") as f:
                self.whistles = json.load(f)

            if len(self.whistles) > 0:
                self.whistle_id = max(w["whistle_id"] for w in self.whistles) + 1
                self.start_time = max(0, self.whistles[-1]["time"] - 2.0)
                print(f"Resuming from {self.start_time}s | Next ID: {self.whistle_id}")
            else:
                self.whistles = []
                self.whistle_id = 0
                self.start_time = 0
        else:
            self.whistles = []
            self.whistle_id = 0
            self.start_time = 0

        # ===============================
        # Layout
        # ===============================
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Video widget
        self.video_widget = QVideoWidget()
        layout.addWidget(self.video_widget)

        # Instructions label
        self.label = QLabel(
            "SPACE=Play | 1=Serve 2=End 3=Other | ←/→ ±5s | A/D ±1s | F=Fullscreen | S=Save | Q=Quit"
        )
        layout.addWidget(self.label)

        # Player
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()

        self.player.setVideoOutput(self.video_widget)
        self.player.setAudioOutput(self.audio_output)
        self.audio_output.setVolume(0.4)

        self.player.setSource(QUrl.fromLocalFile(VIDEO_PATH))

        # Seek to resume position
        self.player.setPosition(int(self.start_time * 1000))
        self.player.mediaStatusChanged.connect(self.handle_media_status)

    def handle_media_status(self, status):
        from PySide6.QtMultimedia import QMediaPlayer
        if status == QMediaPlayer.LoadedMedia and not self._initial_seek_done:
            self.player.setPosition(int(self.start_time * 1000))
            self._initial_seek_done = True
    def mark_whistle(self, whistle_type):
        current_sec = self.player.position() / 1000.0

        whistle = {
            "match_id": MATCH_ID,
            "whistle_id": self.whistle_id,
            "time": round(current_sec, 3),
            "type": whistle_type
        }

        self.whistles.append(whistle)
        self.whistle_id += 1

        print(f"✔ {whistle_type.upper()} @ {whistle['time']}s")

        # Update on-screen label
        self.label.setText(
            f"Last: {whistle_type.upper()} @ {whistle['time']}s | Total: {len(self.whistles)}"
        )
    # ===============================
    # Key Controls
    # ===============================
    def keyPressEvent(self, event):
        key = event.key()

        # -------- PLAY / PAUSE --------
        if key == Qt.Key_Space:
            if self.player.playbackState() == QMediaPlayer.PlayingState:
                self.player.pause()
            else:
                self.player.play()

        # -------- MARK SERVE --------
        elif key == Qt.Key_1:
            self.mark_whistle("serve")

        # -------- MARK RALLY END --------
        elif key == Qt.Key_2:
            self.mark_whistle("rally_end")

        # -------- MARK OTHER --------
        elif key == Qt.Key_3:
            self.mark_whistle("other")

        # -------- SKIPPING --------
        elif key == Qt.Key_Right:
            self.skip(5)

        elif key == Qt.Key_Left:
            self.skip(-5)

        elif key == Qt.Key_D:
            self.skip(1)

        elif key == Qt.Key_A:
            self.skip(-1)

        # -------- FULLSCREEN --------
        elif key == Qt.Key_F:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()

        # -------- SAVE --------
        elif key == Qt.Key_S:
            self.save_annotations()

        # -------- QUIT --------
        elif key == Qt.Key_Q:
            self.save_annotations()
            self.close()

    def skip(self, seconds):
        new_pos = self.player.position() + seconds * 1000
        new_pos = max(0, new_pos)
        self.player.setPosition(new_pos)

    def save_annotations(self):
        with open(OUT_JSON, "w") as f:
            json.dump(self.whistles, f, indent=4)

        print(f"\nSaved {len(self.whistles)} whistles → {OUT_JSON}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WhistleAnnotator()
    window.show()
    sys.exit(app.exec())
