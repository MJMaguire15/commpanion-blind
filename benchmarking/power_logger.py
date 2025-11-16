import psutil, time, json, pathlib
from time import monotonic_ns

class PowerLogger:
    def __init__(self, out="commpanion-blind/benchmarking/runs/power.jsonl", period=1.0):
        self.f = pathlib.Path(out).open("w", buffering=1, encoding="utf-8")
        self.period = period
        self.proc = psutil.Process()

    def loop(self, stop_flag):
        while not stop_flag():
            b = psutil.sensors_battery()
            rec = {
                "t_ns": monotonic_ns(),
                "cpu_percent_proc": self.proc.cpu_percent(None),
                "cpu_percent_sys": psutil.cpu_percent(None),
                "mem_rss_mb": self.proc.memory_info().rss / 1e6,
                "battery_percent": getattr(b, "percent", None),
                "power_plugged": getattr(b, "power_plugged", None),
            }
            self.f.write(json.dumps(rec) + "\n")
            time.sleep(self.period)
    