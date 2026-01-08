# recap.py
import time
from collections import deque

WINDOW_SECONDS = 15
RECAP_INTERVAL = 10

class SemanticRecap:
    def __init__(self, llm):
        self.llm = llm
        self.buffer = deque()
        self.last_recap_time = 0

    def add_text(self, text):
        now = time.time()
        self.buffer.append((now, text))

        while self.buffer and self.buffer[0][0] < now - WINDOW_SECONDS:
            self.buffer.popleft()

    def maybe_recap(self):
        now = time.time()
        if now - self.last_recap_time < RECAP_INTERVAL:
            return None

        window_text = " ".join(t for _, t in self.buffer)
        if len(window_text.strip()) < 20:
            return None

        self.last_recap_time = now
        return self._generate_recap(window_text)

    def _generate_recap(self, text):
        prompt = f"""
You are an assistive summarisation system.
Summarise what happened in the last few seconds of dialogue.
Use simple language.
One sentence only.
Do not quote dialogue.
Focus on meaning, not wording.

Dialogue:
{text}
"""
        return self.llm(prompt).strip()
