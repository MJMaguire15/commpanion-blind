from dataclasses import dataclass, asdict
from time import monotonic_ns, monotonic
import json, pathlib

@dataclass
class Event:
    t_ns: int
    name: str
    data: dict

class Metrics:
    def __init__(self, out_path="commpanion-blind/benchmarking/runs/run.jsonl"):
        p = pathlib.Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.f = p.open("w", buffering=1, encoding="utf-8")

    def emit(self, name, **data):
        self.f.write(json.dumps(asdict(Event(monotonic_ns(), name, data))) + "\n")

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

def lag_seconds_from_src(t_src_ns):
    """How far 'behind real-time' we are right now."""
    return max(0.0, monotonic() - (t_src_ns / 1e9))
