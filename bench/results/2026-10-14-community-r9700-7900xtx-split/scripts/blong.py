"""Long-context decode bench. blong.py LABEL [ENV=VAL ...] [--arg FLAG VAL ...]
One restart; short context: 4 questions; long context: one ~157K prefix read once, then the same 4 questions on it (the
prefix comes from the cache). Per answer: client tok/s and the engine's ms/window + tokens/window (STRATA_DECODE_TIMING)."""
import json, os, re, shutil, statistics, subprocess, sys, time, glob, urllib.request
REPO = os.path.expanduser("~/llm/strata-src/repo"); CFG = f"{REPO}/strata-iq3s-default.json"
BASEF = os.path.expanduser("~/llm/strata-tuned-base-2026-10-14.json"); LOGF = f"{REPO}/strata-iq3s-default.log"
OUT = os.path.expanduser("~/llm/strata-tune-long-2026-10-14.log")
split = None; label = sys.argv[1]; env = {"STRATA_DECODE_TIMING": "1"}; add = []; drop = []; it = iter(sys.argv[2:])
for x in it:
    if x == "--arg": a = next(it); v = next(it); drop.append(a); add += [a, v]
    elif x == "--split": split = next(it)
    else: k, v = x.split("=", 1); env[k] = v
c = json.load(open(BASEF)); c["env"].update(env)
if split: c["layer_split"] = split
for d in drop:
    while d in c["args"]:
        i = c["args"].index(d); del c["args"][i:i + 2]
c["args"] += add
json.dump(c, open(CFG, "w"), indent=2)
r = subprocess.run(["ai", "restart"], capture_output=True, text=True)
if "ready:" not in r.stdout:
    print(f"{label}: FAILED {r.stdout[-300:]}"); shutil.copy(BASEF, CFG); sys.exit(1)
TXT = ""
for f in sorted(glob.glob(f"{REPO}/**/*.py", recursive=True) + glob.glob(f"{REPO}/**/*.md", recursive=True)):
    try: TXT += f"\n### {f}\n" + open(f, errors="ignore").read()
    except Exception: pass
    if len(TXT) > 430000: break
TXT = TXT[:430000]
QS = ["Write a Go function that merges overlapping intervals, with a short explanation.",
      "Explain in detail how a B-tree handles node splits during insertion, with an example.",
      "Describe the request flow of the server code above, from the HTTP handler to the engine, step by step.",
      "Produce a JSON array of 12 objects with fields id, name, email, active for a fake user list."]
def req(msgs, n):
    body = json.dumps({"model": "qwen3.8-flash-next-iq3_s", "messages": msgs, "max_tokens": n, "temperature": 0,
                       "stream": True}).encode()
    first = None; k = 0
    for l in urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8106/v1/chat/completions", body,
                                                           {"Content-Type": "application/json"}), timeout=1800):
        if l.startswith(b"data: {"):
            d = json.loads(l[6:])
            if d.get("choices"):
                ch = d["choices"][0]["delta"]
                if ch.get("content") or ch.get("reasoning_content"):
                    k += 1; first = first or time.time()
    return k / (time.time() - first)
def run(msgs, n=300):
    pos = os.path.getsize(LOGF); tps = req(msgs, n); time.sleep(0.5)
    new = open(LOGF, errors="ignore").read()[pos:] if False else open(LOGF, "rb").read()[pos:].decode(errors="ignore")
    m = re.findall(r"decode timing: \d+ windows, avg T [\d.]+, ([\d.]+) tokens/window, ([\d.]+) ms/window", new)
    tw, ms = (float(m[-1][0]), float(m[-1][1])) if m else (0, 0)
    return tps, ms, tw
req([{"role": "user", "content": "Say hi."}], 30)
res = {}
for ctx in ("short", "157K"):
    rows = []
    for q in QS:
        msgs = [{"role": "user", "content": (TXT + "\n\n" if ctx == "157K" else "") + q}]
        if ctx == "157K" and not rows: run(msgs, 8)            # read the prefix once (not timed)
        rows.append(run(msgs))
    res[ctx] = rows
line = f"{label:<14}" + "".join(
    f" | {ctx}: {statistics.mean(r[0] for r in rows):5.1f} tok/s, {statistics.mean(r[1] for r in rows):5.2f} ms/win, "
    f"{statistics.mean(r[2] for r in rows):4.2f} tok/win [{' '.join(f'{r[0]:.0f}' for r in rows)}]" for ctx, rows in res.items())
print(line, flush=True); open(OUT, "a").write(line + "\n")
shutil.copy(BASEF, CFG)
