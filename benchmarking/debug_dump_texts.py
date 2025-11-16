import json
from itertools import islice
import argparse
from pathlib import Path

def main(run_path: str):
    path = Path(run_path)
    if not path.exists():
        print(f"Run file not found: {path}")
        return

    texts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("name") == "inference_done":
                data = obj.get("data", {})
                texts.append(data.get("text", ""))

    print("num windows:", len(texts))
    print("first 5 non-empty:")
    for t in islice((t for t in texts if t and t.strip()), 5):
        print("  ", t[:120])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="benchmarking/runs/run_stream.jsonl")
    args = p.parse_args()
    main(args.run)
