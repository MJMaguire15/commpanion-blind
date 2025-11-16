import re, editdistance

def _norm(s):
    return re.sub(r"\s+"," ", re.sub(r"[^a-z0-9\s']", " ", s.lower())).strip()

def wer(ref, hyp):
    r, h = _norm(ref).split(), _norm(hyp).split()
    return 0.0 if not r and not h else (editdistance.eval(r, h) / max(1,len(r)))

def cer(ref, hyp):
    r, h = list(_norm(ref)), list(_norm(hyp))
    return 0.0 if not r and not h else (editdistance.eval(r, h) / max(1,len(r)))
