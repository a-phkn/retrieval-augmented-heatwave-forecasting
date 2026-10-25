import json,sys,re
sys.path.insert(0,'.')
from thr_def import THR
exec(open('verify.py').read().split("out=[]")[0])
bad=0
for t in THR:
    r=check(t["doc_id"],t["page"],t["verbatim_excerpt"])
    t["verification"]="VERIFIED (exact text found on cited page)" if r else "FAILED"
    t["verified_modes"]=r
    if not r: bad+=1; print("FAIL",t["quantity"],t["page"],t["verbatim_excerpt"])
json.dump(THR,open("../phase6_thresholds.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print(len(THR),"thresholds;",bad,"failed")
