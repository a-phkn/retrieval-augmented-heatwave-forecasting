import subprocess,re,sys,os
from PIL import Image
pages=[int(x) for x in sys.argv[1:]]
common=set("the of and to in for a is are be by with on as that this heat wave water health during should state district".split())
def score(txt):
    w=re.findall(r"[a-z]+",txt.lower()); 
    return sum(1 for x in w if x in common)
for p in pages:
    n=f"{p:02d}"
    png=f"/tmp/rot{n}.png"
    subprocess.run(["pdftoppm","-r","170","-f",str(p),"-l",str(p),"-png","-singlefile","pdfs/NDMA_HW_GUIDELINES_2019.pdf",f"/tmp/rot{n}"],check=True)
    im=Image.open(png); best=(-1,None,None)
    for ang in (0,90,180,270):
        r=im.rotate(ang,expand=True) if ang else im
        r.save("/tmp/r_try.png")
        out=subprocess.run(["tesseract","/tmp/r_try.png","-","-l","eng","--psm","3"],capture_output=True,text=True).stdout
        s=score(out)
        if s>best[0]: best=(s,ang,out)
        if s>150: break
    open(f"ocr_ndma2019_rot/p{n}.txt","w",encoding="utf-8").write(best[2])
    print(p,"angle",best[1],"score",best[0],flush=True)
