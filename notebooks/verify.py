r"""ตรวจความถูกต้องก่อนไปต่อ (ไม่ใช้โมเดล)
1) นับข้อสอบ nhanes_age_pairwise ทั้งชุด + การกระจายเฉลย A/B
2) สุ่ม 5 ข้อให้ตรวจเฉลยด้วยตา
3) ดูว่า 200 ข้อแรกต่างจากทั้งชุดไหม (อายุห่างเฉลี่ย)
4) ขุด README ของชุดข้อมูล: การแบ่ง train/test + คะแนนที่รายงานไว้"""
import json, random, re, statistics
from collections import Counter
from datasets import load_dataset
from huggingface_hub import hf_hub_download

REPO = "insilicomedicine/longebench"
TASK = "nhanes_age_pairwise"

# ---------- 1) นับทั้งชุด ----------
ds = load_dataset(REPO, "benchmark", streaming=True)
split = list(ds.keys())[0]
items, meta_keys = [], Counter()
for row in ds[split]:
    if row["task"] != TASK:
        continue
    m = json.loads(row["metadata"])
    meta_keys.update(m.keys())
    d = dict(re.findall(r"Participant ([AB])[^:]*:\s*(\d+)", m["follow_up"]))
    a, b = int(d["A"]), int(d["B"])
    items.append(dict(a=a, b=b, gold="A" if a > b else ("B" if b > a else "="),
                      follow_up=m["follow_up"], meta=m))
print(f"[1] {TASK} ทั้งชุด benchmark: {len(items)} ข้อ")
print("    เฉลย A/B/เท่ากัน:", dict(Counter(i["gold"] for i in items)))
print("    ฟิลด์ใน metadata:", dict(meta_keys))

# ---------- 2) สุ่มตรวจด้วยตา ----------
print("\n[2] สุ่ม 5 ข้อ — ตรวจว่า gold = คนที่อายุมากกว่า")
random.seed(0)
for it in random.sample(items, 5):
    print(f"    gold={it['gold']}  <-  {it['follow_up']}")

# ---------- 3) 200 ข้อแรก vs ทั้งชุด ----------
gap = lambda x: abs(x["a"] - x["b"])
first, rest = items[:200], items[200:]
print(f"\n[3] อายุห่างเฉลี่ย: 200 ข้อแรก = {statistics.mean(map(gap, first)):.1f} ปี"
      + (f" | ที่เหลือ {len(rest)} ข้อ = {statistics.mean(map(gap, rest)):.1f} ปี" if rest else " | (มีแค่ 200 ข้อ)"))

# ---------- 4) README ของชุดข้อมูล ----------
path = hf_hub_download(REPO, "README.md", repo_type="dataset")
text = open(path, encoding="utf-8").read()
print(f"\n[4] README ของชุดข้อมูล ({len(text)} ตัวอักษร) — บรรทัดที่เกี่ยวข้อง:")
keys = re.compile(r"nhanes|split|train|test|leak|wave|contamin|LFM|1\.2B|accuracy", re.I)
shown = 0
for line in text.splitlines():
    if keys.search(line) and len(line.strip()) > 3:
        print("    " + line.strip()[:200])
        shown += 1
        if shown >= 40:
            print("    ...(ตัด)"); break
