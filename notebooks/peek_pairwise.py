"""ดูหน้าตาคำถามของโจทย์ pairwise ทุกชุด (1 ข้อต่อชุด) + เฉลย — เพื่อเขียนตัวสลับให้ถูก"""
import json
from datasets import load_dataset

ds = load_dataset("insilicomedicine/longebench", "mini", streaming=True)["eval"]
seen = set()
for row in ds:
    t = row["task"]
    if row["format"] != "pairwise" or t in seen:
        continue
    seen.add(t)
    user = [m for m in row["messages"] if m["role"] == "user"][-1]["content"]
    print("=" * 90)
    print("TASK:", t, "| domain:", row["domain"])
    print("--- ต้นคำถาม (600 ตัวอักษรแรก) ---")
    print(user[:600])
    print("--- ท้ายคำถาม (300 ตัวอักษรสุดท้าย) ---")
    print(user[-300:])
    print("--- เฉลย ---")
    print(json.loads(row["metadata"]).get("follow_up"))
print("\nรวม", len(seen), "ชุด pairwise")
