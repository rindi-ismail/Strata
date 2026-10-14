import json,time,urllib.request,glob,sys
def stream(prompt,n):
    body=json.dumps({"model":"qwen3.8-flash-next-iq3_s","messages":[{"role":"user","content":prompt}],"max_tokens":n,"temperature":0,"stream":True,"stream_options":{"include_usage":True}}).encode()
    t0=time.time(); first=None; toks=0; usage=None
    r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8106/v1/chat/completions",body,{"Content-Type":"application/json"}),timeout=1500)
    for line in r:
        if line.startswith(b"data: {"):
            d=json.loads(line[6:])
            if d.get("usage"): usage=d["usage"]
            if d.get("choices"):
                ch=d["choices"][0]["delta"]
                if ch.get("content") or ch.get("reasoning_content"):
                    toks+=1
                    if first is None: first=time.time()
    return t0,first,time.time(),toks,usage
for i in range(3):
    t0,f,t1,k,u=stream("Write a Go function that merges overlapping intervals, with a short explanation.",300)
    print(f"short run{i}: {k} tok, ttft {f-t0:.2f}s, decode {k/(t1-f):.1f} tok/s")
import os; os.chdir(os.path.expanduser("~/llm/strata-src/repo"))
txt=""
for f in sorted(glob.glob("**/*.py",recursive=True)+glob.glob("**/*.md",recursive=True)):
    try: txt+=f"\n### {f}\n"+open(f,errors="ignore").read()
    except: pass
    if len(txt)>420000: break
txt=txt[:420000]
t0,f,t1,k,u=stream(txt+"\n\nIn two sentences, what is this repository?",120)
pt=u["prompt_tokens"] if u else len(txt)//4
print(f"long: {pt} prompt tok, ttft {f-t0:.1f}s => prefill {pt/(f-t0):.0f} tok/s; decode {k/(t1-f):.1f} tok/s")
