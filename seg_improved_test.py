"""Run improved segmentation on CASIA + MMU samples; build before/after montage."""
import numpy as np, cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import config as cfg
import seg_improved as S

ROOT = Path(r"E:\Strudin\Jamal Nasir Al-khafaji")
OUT = ROOT / "Paper3_Complementarity_PR" / "seg_improved"
OUT.mkdir(parents=True, exist_ok=True)

casia_dir = ROOT / "innovatorsoasis-iris_hind" / "preprocessing_demo_output"
casia = [casia_dir/"1"/"01_original"/"L"/"S5000L00.jpg",
         casia_dir/"1"/"01_original"/"L"/"S5000L02.jpg",
         casia_dir/"2"/"01_original"/"L"/"S5001L00.jpg"]
mmu_dir = ROOT / "Correct_theiss" / "methodology_implementation" / "dataset" / "MMU-Iris-Database"
mmu = sorted(mmu_dir.glob("*/left/*.bmp"))[:2] + sorted(mmu_dir.glob("*/right/*.bmp"))[:1]

samples = [("CASIA", p) for p in casia if p.exists()] + [("MMU", p) for p in mmu if p.exists()]
print(f"{len(samples)} samples")

def to_rgb(g):
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

rows = []
vfs = []
for tag, p in samples:
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if g is None:
        print("skip", p); continue
    r = S.segment_normalise(g)
    ox, oy, _ = r["p_old"]; nx, ny, rp = r["p_new"]
    # overlay: old limbus red (old centre), new limbus green (new centre), new pupil yellow
    ov = to_rgb(r["img"])
    cv2.circle(ov, (ox, oy), r["rl_old"], (0, 0, 255), 2)      # old = red
    cv2.circle(ov, (nx, ny), r["rl_new"], (0, 255, 0), 2)      # new = green
    cv2.circle(ov, (nx, ny), rp, (0, 255, 255), 1)            # pupil = yellow
    # templates to RGB, upscale for visibility
    def tshow(t):
        t3 = to_rgb(t.astype(np.uint8))
        return cv2.resize(t3, (320, 80), interpolation=cv2.INTER_NEAREST)
    rows.append((tag, p.name, ov, tshow(r["t_old"]), tshow(r["t_new"]), r["valid_frac"]))
    vfs.append(r["valid_frac"])
    print(f"{tag:5s} {p.name:14s} rl_old={r['rl_old']:3d} rl_new={r['rl_new']:3d} valid={r['valid_frac']*100:.0f}%")

# assemble montage: each row = [overlay(320x240) | old tmpl(320x80) | new tmpl(320x80)]
cellw = 324
rowh = 248
pad = 10
labelw = 360
W = labelw + 3*cellw + 4*pad
H = len(rows)*(rowh+pad) + pad + 40
canvas = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(canvas)
try:
    fb = ImageFont.truetype("arialbd.ttf", 16); f = ImageFont.truetype("arial.ttf", 14)
except: fb = f = ImageFont.load_default()
# headers
d.text((labelw+pad, 14), "red=OLD limbus  green=NEW  yellow=pupil", fill="#222", font=f)
d.text((labelw+pad+cellw, 60), "OLD template", fill="#B00000", font=fb)
d.text((labelw+pad+2*cellw, 60), "NEW template", fill="#0A7F00", font=fb)
for i,(tag,name,ov,told,tnew,vf) in enumerate(rows):
    y = 40 + i*(rowh+pad)
    # label
    d.text((pad, y+10), f"{tag}", fill="#222", font=fb)
    d.text((pad, y+32), name, fill="#555", font=f)
    d.text((pad, y+58), f"valid iris: {vf*100:.0f}%", fill="#0A7F00", font=f)
    # overlay
    canvas.paste(Image.fromarray(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB)), (labelw+pad, y))
    canvas.paste(Image.fromarray(cv2.cvtColor(told, cv2.COLOR_BGR2RGB)), (labelw+pad+cellw, y+80))
    canvas.paste(Image.fromarray(cv2.cvtColor(tnew, cv2.COLOR_BGR2RGB)), (labelw+pad+2*cellw, y+80))

outp = OUT / "before_after_montage.png"
canvas.save(outp)
print("saved", outp, canvas.size, "mean valid frac", np.mean(vfs))
