"""Wrapper around the fine-tuned classifier.

Long inputs (e.g. tool output) are split into overlapping windows and the
maximum injection score is used, because the model only sees ~128 tokens and
an attack hidden deep in a long document would otherwise be missed.
"""


class Guard:
    def __init__(self, path: str = "guard_model", window_words: int = 80, overlap: int = 20,
                 max_windows: int = 40):
        from transformers import pipeline
        self.clf = pipeline("text-classification", model=path, truncation=True,
                            max_length=128, top_k=None)
        self.window, self.step, self.max_windows = window_words, window_words - overlap, max_windows

    def _chunks(self, text: str):
        words = text.split()
        if len(words) <= self.window:
            return [text]
        chunks = [" ".join(words[i:i + self.window]) for i in range(0, len(words), self.step)]
        return chunks[: self.max_windows]

    def _score_one(self, text: str) -> float:
        res = self.clf(text)
        if res and isinstance(res[0], list):
            res = res[0]
        return next(r["score"] for r in res if r["label"] == "INJECTION")

    def score(self, text: str) -> float:
        text = (text or "").strip()
        if not text:
            return 0.0
        return max(self._score_one(c) for c in self._chunks(text))
