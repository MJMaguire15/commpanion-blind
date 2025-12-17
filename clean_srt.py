import re
from pathlib import Path

def parse_srt_time(t):
    t = t.replace('.', ',')
    hms, ms = t.split(',')
    h, m, s = hms.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s) + int(ms) / 1000.0

def load_srt_blocks(path):
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = raw.splitlines()

    blocks = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue

        # Sequence number
        if lines[i].strip().isdigit():
            i += 1

        # Timing
        if i >= len(lines): break
        if '-->' not in lines[i]:
            i += 1
            continue

        timing = lines[i].strip()
        start_str, end_str = [x.strip() for x in timing.split('-->')]
        start_s = parse_srt_time(start_str)
        end_s   = parse_srt_time(end_str)
        i += 1

        # Gather text lines
        text_lines = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1

        blocks.append((start_s, end_s, text_lines))
    return blocks

def clean_block(block_text_lines):
    # Remove exact duplicates
    dedup = []
    seen = set()
    for line in block_text_lines:
        line_norm = re.sub(r"\s+", " ", line).strip().lower()
        if line_norm not in seen:
            seen.add(line_norm)
            dedup.append(line)
    return dedup

def save_cleaned_srt(blocks, out_path):
    out = []
    for idx, (start_s, end_s, text_lines) in enumerate(blocks, start=1):
        h = lambda t: f"{int(t//3600):02d}:{int((t%3600)//60):02d}:{t%60:06.3f}".replace('.', ',')
        out.append(str(idx))
        out.append(f"{h(start_s)} --> {h(end_s)}")
        for line in text_lines:
            out.append(line)
        out.append("")  # blank separator

    Path(out_path).write_text("\n".join(out), encoding="utf-8")
    print(f"Saved cleaned SRT to {out_path}")

def main():
    src = input("Enter original SRT path: ").strip('"')
    out = src.replace(".srt", "_cleaned.srt")

    blocks = load_srt_blocks(src)
    cleaned = [(s,e,clean_block(txt)) for (s,e,txt) in blocks]
    save_cleaned_srt(cleaned, out)

if __name__ == "__main__":
    main()
