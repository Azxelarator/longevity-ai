"""ขั้นที่ 3: นับว่าในชุด 'mini' มีโจทย์กี่แบบ แบบละกี่ข้อ (streaming)"""
from collections import Counter
from datasets import load_dataset

ds = load_dataset("insilicomedicine/longebench", "mini", streaming=True)["eval"]

by_task = Counter()
info = {}
for row in ds:
    t = row["task"]
    by_task[t] += 1
    info[t] = (row["domain"], row["format"], row["metric"])

print(f"รวม {sum(by_task.values())} ข้อ, {len(by_task)} แบบโจทย์\n")
print(f"{'task':40} {'domain':15} {'format':12} {'metric':12} {'n':>5}")
print("-" * 88)
for t, n in sorted(by_task.items(), key=lambda x: info[x[0]]):
    d, f, m = info[t]
    print(f"{t:40} {d:15} {f:12} {m:12} {n:>5}")
