r"""ดูหน้าตาข้อมูลการเสียชีวิต (NHANES mortality) ทุกแบบ ใน benchmark + extra"""
import json, re
from collections import defaultdict
from datasets import load_dataset

REPO = "insilicomedicine/longebench"
age_seqn, found = set(), defaultdict(list)
for cfg in ["benchmark", "extra"]:
    ds = load_dataset(REPO, cfg, streaming=True)
    for row in ds[list(ds.keys())[0]]:
        t = row["task"]
        meta = json.loads(row["metadata"]) if row["metadata"] else {}
        fu = str(meta.get("follow_up", ""))
        if t == "nhanes_age_regression":
            age_seqn.update(re.findall(r"SEQN (\d+)", fu))
        elif t.startswith("nhanes_mortality"):
            found[(cfg, t)].append((row, fu))

mort_seqn = set()
for (cfg, t), rows in found.items():
    row, fu = rows[0]
    user = [m for m in row["messages"] if m["role"] == "user"][-1]["content"]
    s = set(x for _, f in rows for x in re.findall(r"SEQN (\d+)", f))
    mort_seqn |= s
    print("=" * 90)
    print(f"{cfg} / {t}: {len(rows)} ข้อ | SEQN ไม่ซ้ำ {len(s)} คน")
    print("--- คำถาม (700 ตัวแรก) ---\n" + user[:700])
    print("--- ท้ายคำถาม (250) ---\n" + user[-250:])
    print("--- เฉลย 3 ข้อแรก ---")
    for _, f in rows[:3]:
        print("  " + f[:250])
print("=" * 90)
print(f"คนในข้อมูล mortality ทั้งหมด: {len(mort_seqn)} | คนใน age_regression: {len(age_seqn)} | ซ้ำกัน: {len(mort_seqn & age_seqn)}")
