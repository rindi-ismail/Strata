"""Decode profile vs context length. Usage: prof.py LABEL [ENV=VAL ...] [--arg X ...]
Restarts the server with the profiling env (plus extras), runs ~1K, 64K, 157K prompts (300 decode tokens each), and prints
the engine log lines each request produced (decode timing, split timing, per-layer verify profile)."""
import json, os, random, shutil, subprocess, sys, time, glob, urllib.request
REPO = os.path.expanduser("~/llm/strata-src/repo"); CFG = f"{REPO}/strata-iq3s-default.json"
BASEF = os.path.expanduser("~/llm/strata-tuned-base-2026-10-14.json"); LOGF = f"{REPO}/strata-iq3s-default.log"
label = sys.argv[1]; env = {}; add = []; drop = []; it = iter(sys.argv[2:])
for x in it:
    if x == "--arg": a = next(it); v = next(it); drop.append(a); add += [a, v]
    elif x == "--noprof": env["__noprof"] = "1"
    else: k, v = x.split("=", 1); env[k] = v
prof = env.pop("__noprof", None) is None
c = json.load(open(BASEF))
if prof: c["env"].update({"STRATA_VERIFY_PROFILE": "1", "STRATA_DECODE_TIMING": "1", "STRATA_SPLIT_TIMING": "1"})
c["env"].update(env)
for d in drop:
    while d in c["args"]:
        i = c["args"].index(d); del c["args"][i:i + 2]
c["args"] += add
json.dump(c, open(CFG, "w"), indent=2)
r = subprocess.run(["ai", "restart"], capture_output=True, text=True)
assert "ready:" in r.stdout, r.stdout[-400:]
TXT = ""
for f in sorted(glob.glob(f"{REPO}/**/*.py", recursive=True) + glob.glob(f"{REPO}/**/*.md", recursive=True)):
    try: TXT += f"\n### {f}\n" + open(f, errors="ignore").read()
    except Exception: pass
    if len(TXT) > 430000: break
def req(p, n):
    body = json.dumps({"model": "qwen3.8-flash-next-iq3_s", "messages": [{"role": "user", "content": p}], "max_tokens": n,
                       "temperature": 0, "stream": True}).encode()
    t0 = time.time(); first = None; k = 0
    for l in urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8106/v1/chat/completions", body,
                                                           {"Content-Type": "application/json"}), timeout=1800):
        if l.startswith(b"data: {"):
            d = json.loads(l[6:])
            if d.get("choices"):
                ch = d["choices"][0]["delta"]
                if ch.get("content") or ch.get("reasoning_content"):
                    k += 1; first = first or time.time()
    return k / (time.time() - first)
ASK = "\n\nWrite a detailed, structured technical summary of the code above: modules, data flow, and key functions."
req("Say hi.", 30)
for name, chars in [("1K", 4000), ("64K", 250000), ("157K", 430000)]:
    pos = os.path.getsize(LOGF)
    tps = req(f"[{label} {random.random():.10f}]\n" + TXT[:chars] + ASK, 300)
    time.sleep(1)
    with open(LOGF, errors="ignore") as f:
        f.seek(pos); new = f.read()
    keep = [l for l in new.splitlines() if any(s in l for s in ("prompt ", "decode", "stage ", "GDN layers", "KV streaming", "hit rate"))]
    print(f"=== {label} {name}: client decode {tps:.1f} tok/s"); print("\n".join(keep)[:6000], flush=True)
shutil.copy(BASEF, CFG)
