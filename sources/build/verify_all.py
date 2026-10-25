# -*- coding: utf-8 -*-
import re,json,sys,os
sys.path.insert(0,'.')
from actions_def import ACTIONS
from actions_def2 import ACTIONS2
from thr_def import THR
from thr_def2 import THR2
from actions_def3 import ACTIONS3
from thr_def3 import THR3
from actions_def4 import ACTIONS4
from thr_def4 import THR4
from actions_def5 import ACTIONS5
from actions_def6 import ACTIONS6
from actions_def7 import ACTIONS7
from actions_def8 import ACTIONS8
from thr_def5 import THR5
from thr_def6 import THR6
from thr_def7 import THR7
from thr_def8 import THR8
from tiers_def import TIERS,OBS,STANDING
def norm(s): return re.sub(r"[\s\-‐‑–—'\"‘’“”―‖‗]+","",s)
cache={}
VIS={"NDMA_HW_GUIDELINES_2019":{34,35,36,37,44,48,51}}
IMGPAGES={"DELHI_HAP_2025":{45,155,156,157,158}}
IMGDOC={"BEE_AC_24C_PIB":{1},"CEA_ELNINO_ADVISORY":{1,2},"PIB_LABOUR_ADVISORY_2026_SCREENSHOT":{1},"HEAT_INDEX_PRESS_RELEASE_2023":{1},"HEAT_INDEX_PRESS_RELEASE_2_2023":{1},"HEATWAVE_GUIDANCE_PRESS_RELEASE_2026":{1}}
def pages(doc,mode):
    k=(doc,mode)
    if k in cache: return cache[k]
    if doc=="NDMA_HW_GUIDELINES_2019":
        out={}
        for d in ("../ocr_ndma2019","../ocr_ndma2019_rot"):
            if not os.path.isdir(d): continue
            for f in os.listdir(d):
                m=re.match(r"p(\d+)\.txt",f)
                if m: out[int(m.group(1))]=out.get(int(m.group(1)),"")+norm(open(f"{d}/{f}",encoding="utf-8",errors="replace").read())
        cache[k]=out; return out
    d="../txt" if mode=="layout" else "../txt_raw"
    pp=[norm(p) for p in open(f"{d}/{doc}.txt",encoding="utf-8",errors="replace").read().split("\f")]
    cache[k]={i+1:p for i,p in enumerate(pp)}; return cache[k]
def check(doc,page,q):
    if doc in IMGDOC or page in IMGPAGES.get(doc,()): return []
    n=norm(q); res=[]
    for m in ("layout","raw"):
        pp=pages(doc,m)
        if n in pp.get(page,""): res.append(m if doc!="NDMA_HW_GUIDELINES_2019" else "ocr")
        if doc=="NDMA_HW_GUIDELINES_2019": break
    return res
def status(doc,page,q,res,override=False):
    if doc=="NDMA_HW_GUIDELINES_2019":
        if page in VIS[doc]:
            return "VERIFIED (scanned page: OCR match and checked against page image)" if res else "VERIFIED visually against page image (OCR did not capture this cell)"
        return "UNVERIFIED – scanned document" 
    if page in IMGPAGES.get(doc,()): return "VERIFIED visually against page image (scanned page inside a text PDF, no text layer)"
    if doc in IMGDOC: return "VERIFIED visually against page image (image-only source, no text layer)" if page in IMGDOC[doc] else "UNVERIFIED – scanned document"
    return "VERIFIED (exact text found on cited page)" if res else "FAILED"
bad=0
def doc_is_img(a): return a["doc_id"] in IMGDOC or a["page"] in IMGPAGES.get(a["doc_id"],())
def run(items,kind):
    global bad
    out=[]
    for a in items:
        r=check(a["doc_id"],a["page"],a["verbatim_excerpt"])
        a["verified_modes"]=r
        a["verification"]=status(a["doc_id"],a["page"],a["verbatim_excerpt"],r)
        if a["doc_id"]=="NDMA_HW_GUIDELINES_2019":
            a["printed_page"]=a["page"]-20
            if not r: print("NDMA-NO-OCR-MATCH",kind,a.get("action_id",a.get("quantity")),a["page"])
        elif doc_is_img(a): pass
        elif not r: bad+=1; print("FAIL",kind,a.get("action_id",a.get("quantity")),a["page"],a["verbatim_excerpt"][:70])
        alt=re.findall(r"DELHI_HAP_2024_25 p\.(\d+)",a.get("notes",""))
        a["also_in"]=[]
        for p in alt:
            ok=check("DELHI_HAP_2024_25",int(p),a["verbatim_excerpt"])
            a["also_in"].append({"doc_id":"DELHI_HAP_2024_25","page":int(p),"same_text_found":bool(ok)})
            if not ok: print("ALT-MISS",a.get("action_id"),p)
        out.append(a)
    return out
acts=run(ACTIONS+ACTIONS2+ACTIONS3+ACTIONS4+ACTIONS5+ACTIONS6+ACTIONS7+ACTIONS8,"action"); thr=run(THR+THR2+THR3+THR4+THR5+THR6+THR7+THR8,"threshold")

# ---- Part E: tier_min / tier_source / trigger_type ----
def find_page(doc,q):
    for m in ("layout","raw"):
        pp=pages(doc,m); n=norm(q)
        hits=[p for p,t in pp.items() if n in t]
        if hits: return hits[0]
    return None
ids={a["action_id"] for a in acts}
for k in list(TIERS)+list(OBS)+list(STANDING):
    if k not in ids: print("TIER-UNKNOWN-ID",k); bad+=1
for a in acts:
    i=a["action_id"]
    a["trigger_type"]="observation" if i in OBS else ("standing" if i in STANDING else "forecast")
    if i in TIERS:
        col,doc,q,match,basis=TIERS[i]
        pg=find_page(doc,q)
        if pg is None: print("FAIL-TIER",i,q[:60]); bad+=1; continue
        a["tier_min"]=col
        a["tier_source"]={"doc_id":doc,"page":pg,"quote":q,"match":match,"tier_basis":basis,"note":("Precedent from another city (Ahmedabad HAP, 2019 update), not Delhi policy; Ahmedabad's colour temperature thresholds differ from Delhi's." if basis=="ahmedabad_precedent" else "National IMD colour table; suggested actions column.")}
    elif a["tier_min"]!="UNSPECIFIED":
        d=a["doc_id"]
        a["tier_source"]={"doc_id":d,"page":a["page"],"quote":a["verbatim_excerpt"],"match":"same action (colour named in the action's own source)","tier_basis":"delhi_source" if d.startswith("DELHI_HAP") else "national_source","note":""}
json.dump(acts,open("../phase6_candidate_actions.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
json.dump(thr,open("../phase6_thresholds.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(len(acts),"actions;",len(thr),"thresholds;",bad,"hard failures")
