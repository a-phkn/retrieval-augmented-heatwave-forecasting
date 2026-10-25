# -*- coding: utf-8 -*-
import re,json,sys
sys.path.insert(0,'.')
from actions_def import ACTIONS
def norm(s): return re.sub(r"[\s\-‐‑–—'\"‘’“”―‖‗]+","",s)
cache={}
def pages(doc,mode):
    k=(doc,mode)
    if k not in cache:
        d="../txt" if mode=="layout" else "../txt_raw"
        cache[k]=[norm(p) for p in open(f"{d}/{doc}.txt",encoding="utf-8",errors="replace").read().split("\f")]
    return cache[k]
def check(doc,page,q):
    n=norm(q)
    res=[]
    for m in ("layout","raw"):
        pp=pages(doc,m)
        if page-1<len(pp) and n in pp[page-1]: res.append(m)
    return res
out=[];bad=0
for a in ACTIONS:
    r=check(a["doc_id"],a["page"],a["verbatim_excerpt"])
    a["verified_modes"]=r
    a["verification"]="VERIFIED (exact text found on cited page)" if r else "FAILED"
    if not r: bad+=1; print("FAIL",a["action_id"])
    # cross-check 'also' pages in 2024-25
    alt=re.findall(r"DELHI_HAP_2024_25 p\.(\d+)",a["notes"])
    a["also_in"]=[]
    for p in alt:
        ok=check("DELHI_HAP_2024_25",int(p),a["verbatim_excerpt"])
        a["also_in"].append({"doc_id":"DELHI_HAP_2024_25","page":int(p),"same_text_found":bool(ok)})
        if not ok: print("ALT-MISS",a["action_id"],p)
    out.append(a)
json.dump(out,open("../phase6_candidate_actions.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(len(out),"actions;",bad,"failed")
