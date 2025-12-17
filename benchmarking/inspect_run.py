import json
import argparse
from collections import Counter

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="Path to run_stream.jsonl")
    args = ap.parse_args()

    counts = Counter()
    window_examples = []
    infer_examples = []

    with open(args.run, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            evt = json.loads(line)
            name = evt.get("name")
            counts[name] += 1

            if name == "window_ready" and len(window_examples) < 5:
                window_examples.append(evt)
            if name == "inference_done" and len(infer_examples) < 5:
                infer_examples.append(evt)

    print("=== Event counts ===")
    for k, v in counts.most_common():
        print(f"{k:20s} {v}")

    print("\n=== Sample window_ready events (up to 5) ===")
    for e in window_examples:
        data = e.get("data", {})
        print(f"t0={data.get('t0')}  t1={data.get('t1')}  n={data.get('n')}")

    print("\n=== Sample inference_done events (up to 5) ===")
    for e in infer_examples:
        data = e.get("data", {})
        print(f"text={repr(data.get('text'))}  "
              f"infer_dur_s={data.get('infer_dur_s')}  "
              f"cap_to_infer_s={data.get('cap_to_infer_s')}")

if __name__ == "__main__":
    main()
