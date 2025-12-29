# subtitle_overlay.py
import sys
import time
from PyQt6.QtWidgets import QApplication, QLabel, QWidget
from PyQt6.QtCore import Qt

# ============================================================
# CONFIG — EDIT ONLY THIS SECTION
# ============================================================

# ---- SIMPLE TOGGLES ----
ELDERLY_MODE = True
ROLLING_CAPTIONS = True
STABILISATION_ENABLED = True   # <<< toggle ON / OFF here

# ---- Text constraints ----
MAX_LINES = 2
CHARS_PER_LINE = 42
ELLIPSIS = "…"

# ---- Stabilisation params ----
STABLE_TIME_SEC = 1.0
MIN_STABLE_CHARS = 16
PREFIX_MATCH_RATIO = 0.7
MIN_UPDATE_INTERVAL = 0.0

# ---- Visuals ----
DIM_ENABLED = True
DIM_HEIGHT_RATIO = 0.25
DIM_RGBA_ALPHA = 35

SUBTITLE_FONT_SIZE = 42
SUBTITLE_BG_ALPHA = 170
SUBTITLE_PADDING_X = 28
SUBTITLE_PADDING_Y = 14
SUBTITLE_BOTTOM_MARGIN = 90

# ============================================================
# DO NOT EDIT BELOW
# ============================================================

class BlackBaseLayer(QWidget):
    def __init__(self, geom):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setStyleSheet("background-color: black;")
        self.setGeometry(geom)
        self.show()


class LowerDimLayer(QWidget):
    def __init__(self, geom):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )
        self.setStyleSheet(f"background-color: rgba(0,0,0,{DIM_RGBA_ALPHA});")
        h = int(geom.height() * DIM_HEIGHT_RATIO)
        self.setGeometry(
            geom.x(),
            geom.y() + geom.height() - h,
            geom.width(),
            h
        )
        self.show()


class SubtitleOverlay(QLabel):
    def __init__(self, geom):
        super().__init__()

        self.geom = geom
        self.last_update = 0.0

        # rolling + stabilisation state
        self.prev_line = ""
        self.pending_line = ""
        self.pending_since = 0.0

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.setStyleSheet(f"""
            QLabel {{
                color: white;
                font-size: {SUBTITLE_FONT_SIZE}px;
                font-weight: bold;
                background-color: rgba(0,0,0,{SUBTITLE_BG_ALPHA});
                padding: {SUBTITLE_PADDING_Y}px {SUBTITLE_PADDING_X}px;
                border-radius: 14px;
            }}
        """)

        self.setText("")
        self._reposition()
        self.show()

    # ---------------- layout ----------------
    def _reposition(self):
        self.adjustSize()
        self.resize(self.sizeHint())
        x = self.geom.x() + (self.geom.width() - self.width()) // 2
        y = self.geom.y() + self.geom.height() - self.height() - SUBTITLE_BOTTOM_MARGIN
        self.move(x, y)

    # ---------------- helpers ----------------
    def _prefix_ratio(self, a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        i = 0
        for ca, cb in zip(a, b):
            if ca == cb:
                i += 1
            else:
                break
        return i / max(1, min(len(a), len(b)))

    def _format_two_lines(self, text: str):
        words = text.split()
        lines, current = [], ""

        for w in words:
            if len(current) + len(w) + 1 <= CHARS_PER_LINE:
                current = (current + " " + w).strip()
            else:
                lines.append(current)
                current = w
            if len(lines) == MAX_LINES:
                break

        if len(lines) < MAX_LINES and current:
            lines.append(current)

        if len(words) > len(" ".join(lines).split()):
            max_len = CHARS_PER_LINE - len(ELLIPSIS)
            lines[-1] = lines[-1][:max_len].rstrip(" .") + ELLIPSIS

        while len(lines) < 2:
            lines.insert(0, "")

        return lines[0], lines[1]

    # ---------------- API ----------------
    def set_text(self, text: str):
        text = text.strip()
        if not text:
            return

        now = time.time()
        if now - self.last_update < MIN_UPDATE_INTERVAL:
            return
        self.last_update = now

        top, bottom = self._format_two_lines(text)

        if STABILISATION_ENABLED:
            if not self.pending_line:
                self.pending_line = bottom
                self.pending_since = now
            else:
                sim = self._prefix_ratio(self.pending_line, bottom)
                if sim >= PREFIX_MATCH_RATIO:
                    if (now - self.pending_since >= STABLE_TIME_SEC
                            and len(bottom) >= MIN_STABLE_CHARS):
                        self.prev_line = bottom
                        self.pending_line = ""
                        self.pending_since = 0.0
                else:
                    self.pending_line = bottom
                    self.pending_since = now
        else:
            self.prev_line = bottom

        if ELDERLY_MODE and ROLLING_CAPTIONS:
            display = f"{self.prev_line}\n{bottom}".strip()
        else:
            display = text

        self.setText(display)
        self._reposition()


# ---------------- standalone test ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)

    screens = QApplication.screens()
    if len(screens) < 2:
        print("❌ A3 must be in EXTENDED display mode.")
        sys.exit(1)

    geom = screens[-1].geometry()

    base = BlackBaseLayer(geom)
    if DIM_ENABLED:
        dim = LowerDimLayer(geom)

    sub = SubtitleOverlay(geom)
    # sub.set_text(
    #     "THIS IS THE WAY IN THE PROCESS OF WHAT YOU'VE HAD "
    #     "THIS IS THE SMALL SIZE OF THE SUN"
    # )

    sys.exit(app.exec())
