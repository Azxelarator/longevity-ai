r"""ขั้นที่ 3 เตรียม: ดูหน้าตา nhanes_age_regression (ใช้สร้าง baseline)
+ นับจำนวนข้อ + ดูว่ามีคน (SEQN) ซ้ำกับข้อสอบ pairwise ไหม + ดู config 'extra' คร่าวๆ"""
import json, re
from collections import Counter
from datasets import load_dataset

REPO = "insilicomedicine/longebench"
ds = load_dataset(REPO, "benchmark", streaming=True)
split = list(ds.keys())[0]

reg, pair_seqn, first = [], set(), None
for row in ds[split]:
    t = row["task"]
    meta = json.loads(row["metadata"]) if row["metadata"] else {}
    fu = meta.get("follow_up") or ""
    if t == "nhanes_age_regression":
        reg.append(fu)
        if first is None:
            first = row
    elif t == "nhanes_age_pairwise":
        pair_seqn.update(re.findall(r"SEQN (\d+)", fu))

print(f"nhanes_age_regression: {len(reg)} ข้อ")
if first:
    user = [m for m in first["messages"] if m["role"] == "user"][-1]["content"]
    print("--- ตัวอย่างคำถาม (1200 ตัวแรก) ---\n" + user[:1200])
    print("--- ท้ายคำถาม (300) ---\n" + user[-300:])
    print("--- เฉลย ---\n" + str(json.loads(first["metadata"] or "{}").get("follow_up")))
    print("--- metadata ทั้งหมด ---\n" + str(first["metadata"])[:500])

reg_seqn = set(s for fu in reg for s in re.findall(r"SEQN (\d+)", fu))
print(f"\nSEQN ใน regression: {len(reg_seqn)} คน | ใน pairwise: {len(pair_seqn)} คน | ซ้ำกัน: {len(reg_seqn & pair_seqn)} คน")

# config extra มีอะไร
ex = load_dataset(REPO, "extra", streaming=True)
es = list(ex.keys())[0]
c = Counter()
for i, row in enumerate(ex[es]):
    c[row["task"]] += 1
    if i >= 20000:
        break
print(f"\nconfig 'extra' (split {es}) — task ที่เจอ (นับได้ถึง 20000 แถว):")
for t, n in c.most_common():
    print(f"  {t:45} {n}")
