"""Mk.2.1: แก้ความเอียง B โดยไม่ต้องเทรนใหม่ — ใช้ผลที่บันทึกไว้แล้ว
กติกา: ถามสองรอบ (ปกติ + สลับ) ถ้าคำตอบสลับตามกัน = เชื่อ, ถ้าไม่ = 'ไม่แน่ใจ'"""
import csv
from collections import Counter

rows = list(csv.DictReader(open("results/eval_swap.csv", encoding="utf-8")))
n = len(rows)

# วิธีเดิม: ถามรอบเดียว
base_acc = sum(r["pred"] == r["gold"] for r in rows) / n

# วิธีใหม่: เชื่อเฉพาะข้อที่ consistent
sure = [r for r in rows if r["consistent"] == "1"]
unsure = [r for r in rows if r["consistent"] != "1"]
sure_acc = sum(r["pred"] == r["gold"] for r in sure) / len(sure)
coverage = len(sure) / n

# ข้อที่ไม่แน่ใจ เอียงไปทางไหน
unsure_pattern = Counter(f'{r["pred"]}/{r["pred_swapped"]}' for r in unsure)

# อายุต่างกันเยอะ vs น้อย — ข้อไหนยาก
def gap(r): return abs(int(r["age_A"]) - int(r["age_B"]))
print(f"ทั้งหมด {n} ข้อ\n")
print(f"วิธีเดิม (ถามรอบเดียว)    : ถูก {base_acc:.1%} ตอบทุกข้อ")
print(f"วิธีใหม่ (ถาม 2 รอบ)      : ถูก {sure_acc:.1%} แต่ตอบแค่ {coverage:.0%} ของข้อ ({len(sure)} ข้อ)")
print(f"ข้อที่ไม่แน่ใจ ({len(unsure)} ข้อ) ตอบแบบ ปกติ/สลับ : {dict(unsure_pattern)}\n")

print("ความแม่นตามช่วงอายุที่ห่างกัน (เฉพาะข้อที่แน่ใจ):")
for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 200)]:
    b = [r for r in sure if lo <= gap(r) < hi]
    if b:
        a = sum(r["pred"] == r["gold"] for r in b) / len(b)
        print(f"  ห่าง {lo:>2}-{hi:<3} ปี : {a:.1%}  ({len(b)} ข้อ)")
