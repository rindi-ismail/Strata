"""A/B sweep for the asymmetric pair (R9700 stage 0, XTX stage 1). One fresh restart per arm.
Decode: 4 prompts x 2 rounds after warm-up (tok/s from the stream). Prefill: one fresh ~32K-token prompt (nonce first,
so nothing is reused), tok/s = prompt_tokens / ttft. Greedy text hash per prompt to flag arms whose answers change."""
import json, time, urllib.request, subprocess, sys, os, shutil, statistics, glob, hashlib, random
REPO = os.path.expanduser("~/llm/strata-src/repo"); CFG = f"{REPO}/strata-iq3s-default.json"
BASEF = os.path.expanduser("~/llm/strata-tuned-base-2026-10-14.json")
BASE = json.load(open(BASEF))
LOG = os.path.expanduser("~/llm/strata-tune-asym-2026-10-14.log")
PROMPTS = ["Write a Go function that merges overlapping intervals, with a short explanation.",
 "Explain in detail how a B-tree handles node splits during insertion, with an example.",
 "Rewrite this to use early returns and add error wrapping:\nfunc f(a int) (int, error) {\n if a > 0 {\n  if a < 100 {\n   v, err := g(a)\n   if err == nil {\n    return v, nil\n   } else {\n    return 0, err\n   }\n  } else {\n   return 0, errors.New(\"big\")\n  }\n } else {\n  return 0, errors.New(\"neg\")\n }\n}",
 "Produce a JSON array of 12 objects with fields id, name, email, active for a fake user list."]
TXT = ""
for f in sorted(glob.glob(f"{REPO}/serve/**/*.py", recursive=True)):
    TXT += f"\n### {os.path.basename(f)}\n" + open(f, errors="ignore").read()
    if len(TXT) > 125000: break
TXT = TXT[:125000]

def stream(p, n):
    body = json.dumps({"model": "qwen3.8-flash-next-iq3_s", "messages": [{"role": "user", "content": p}], "max_tokens": n,
                       "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}).encode()
    t0 = time.time(); first = None; k = 0; usage = None; out = []
    r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8106/v1/chat/completions", body,
                                                      {"Content-Type": "application/json"}), timeout=1500)
    for l in r:
        if l.startswith(b"data: {"):
            d = json.loads(l[6:])
            if d.get("usage"): usage = d["usage"]
            if d.get("choices"):
                c = d["choices"][0]["delta"]
                s = (c.get("content") or "") + (c.get("reasoning_content") or "")
                if s:
                    k += 1; out.append(s)
                    if first is None: first = time.time()
    t1 = time.time()
    return k / (t1 - first), first - t0, usage, "".join(out)

def arm(name, split=None, add=(), env=None, drop=()):
    c = json.loads(json.dumps(BASE))
    if split: c["layer_split"] = split
    a = c["args"]
    for d in drop:
        while d in a:
            i = a.index(d); del a[i:i + 2]
    a += list(add)
    if env: c["env"].update(env)
    json.dump(c, open(CFG, "w"), indent=2)
    r = subprocess.run(["ai", "restart"], capture_output=True, text=True)
    if "ready:" not in r.stdout:
        return f"{name:<12} FAILED {r.stdout[-300:]!r}"
    try:
        for p in PROMPTS[:2]: stream(p, 120)
        rounds = [[stream(p, 300) for p in PROMPTS] for _ in range(2)]
        per = [statistics.mean(x[0] for x in col) for col in zip(*rounds)]
        hashes = "".join(hashlib.md5(x[3].encode()).hexdigest()[:4] for x in rounds[0])
        nonce = f"[{name} {random.random():.12f}]\n"
        _, ttft, u, _ = stream(nonce + TXT + "\n\nIn two sentences, what is this code?", 60)
        pt = u["prompt_tokens"] if u else 0
        return (f"{name:<12} decode {statistics.mean(per):5.1f}  per {' '.join(f'{x:5.1f}' for x in per)}  "
                f"prefill {pt/ttft:6.0f} tok/s ({pt} tok)  text {hashes}")
    except Exception as e:
        return f"{name:<12} ERROR {e!r}"

GFX1151_DECODE = {k: "1" for k in ["STRATA_Q8_PACKED", "STRATA_Q6_PACKED", "STRATA_MMVF_ROWS", "STRATA_ATTN_LANECELL",
                                    "STRATA_EXPERT_V2", "STRATA_TSUM", "STRATA_LFUSE", "STRATA_GDN_SPLIT", "STRATA_QFUSE",
                                    "STRATA_PLE_BATCH", "STRATA_SH_STREAM"]}
ARMS = {
    "default":     dict(),
    "trim":        dict(env={"STRATA_STAGE_TRIM": "1"}),
    "trim-s27":    dict(split="27", env={"STRATA_STAGE_TRIM": "1"}),
    "trim-s28":    dict(split="28", env={"STRATA_STAGE_TRIM": "1"}),
    "s27":         dict(split="27"),
    "mtpbatch":    dict(env={"STRATA_SPLIT_MTP_BATCH": "1"}),
    "shstream":    dict(env={"STRATA_SH_STREAM": "1"}),
    "g1151dec":    dict(env=GFX1151_DECODE),
    "lt2":         dict(env={"STRATA_HIPBLASLT_TUNING": f"{REPO}/tools/hip/gfx1201-hipblaslt-100202.txt:{REPO}/tools/hip/gfx1100-hipblaslt-100202.txt"}),
    "default-end": dict(),
    "lt2-confirm": dict(),
}
sel = sys.argv[1:] or list(ARMS)
with open(LOG, "a") as out:
    out.write(f"# {time.strftime('%F %T')} asym sweep: {' '.join(sel)}\n")
    for name in sel:
        line = arm(name, **ARMS[name])
        print(line, flush=True); out.write(line + "\n"); out.flush()
shutil.copy(BASEF, CFG)
subprocess.run(["ai", "restart"], capture_output=True, text=True)
print("restored base config", flush=True)
