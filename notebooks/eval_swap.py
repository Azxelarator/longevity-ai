"""Mk.2: ทดสอบว่าโมเดลอ่านผลเลือดจริง หรือแค่ชอบตัวอักษร
แต่ละข้อถาม 2 รอบ: ปกติ กับ สลับข้อมูล A<->B (เฉลยต้องสลับตาม)
ถ้าโมเดลเข้าใจจริง -> คำตอบต้องสลับตาม (consistent)"""
import csv, json, re, time
from collections import Counter
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "LiquidAI/LFM2-1.2B-Longevity"
TASK = "nhanes_age_pairwise"
CONFIG = "benchmark"
LIMIT = 200

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16 if device == "cuda" else torch.float32
).to(device).eval()

def gold_answer(meta):
    ages = dict(re.findall(r"Participant ([AB])[^:]*:\s*(\d+)", json.loads(meta)["follow_up"]))
    return ("A" if int(ages["A"]) > int(ages["B"]) else "B"), ages

def swap_ab(text):
    """สลับข้อมูลของ Participant A กับ B โดยคงลำดับหัวข้อไว้"""
    pre, rest = text.split("Participant A:", 1)
    a, b = rest.split("Participant B:", 1)
    return f"{pre}Participant A:{b.rstrip()}\n\nParticipant B:{a.rstrip()}"

def ask(messages):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=5, do_sample=False)
    reply = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return next((c for c in reply.upper() if c in "AB"), "?")

ds = load_dataset("insilicomedicine/longebench", CONFIG, streaming=True)
split = list(ds.keys())[0]
rows_out, n = [], 0
shown = False
t0 = time.time()
for row in ds[split]:
    if row["task"] != TASK:
        continue
    gold, ages = gold_answer(row["metadata"])
    msgs = row["messages"]
    user_i = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    swapped = [dict(m) for m in msgs]
    swapped[user_i]["content"] = swap_ab(msgs[user_i]["content"])
    if not shown:  # โชว์ตัวอย่างข้อสลับ 1 ข้อ ให้ตาเช็กว่าสลับถูก
        print("---- ตัวอย่างข้อสลับ (ตัดให้สั้น) ----")
        print(swapped[user_i]["content"][:700], "\n--------------------------------")
        shown = True
    gold_s = "B" if gold == "A" else "A"
    p1, p2 = ask(msgs), ask(swapped)
    n += 1
    rows_out.append(dict(i=n, age_A=ages["A"], age_B=ages["B"], gold=gold, pred=p1,
                         gold_swapped=gold_s, pred_swapped=p2,
                         consistent=int(p1 != p2 and "?" not in (p1, p2))))
    if n % 20 == 0:
        print(f"{n} ข้อ... ({time.time()-t0:.0f}s)")
    if n >= LIMIT:
        break

acc1 = sum(r["pred"] == r["gold"] for r in rows_out) / n
acc2 = sum(r["pred_swapped"] == r["gold_swapped"] for r in rows_out) / n
cons = sum(r["consistent"] for r in rows_out) / n
both = sum(r["pred"] == r["gold"] and r["pred_swapped"] == r["gold_swapped"] for r in rows_out) / n
letters = Counter([r["pred"] for r in rows_out] + [r["pred_swapped"] for r in rows_out])

print(f"\n===== ผล ({n} ข้อ, {CONFIG}) =====")
print(f"Accuracy ปกติ        : {acc1:.1%}")
print(f"Accuracy สลับ        : {acc2:.1%}")
print(f"สลับคำตอบตาม (consistent): {cons:.1%}   <- ยิ่งสูงยิ่งดี")
print(f"ถูกทั้งสองรอบ (เข้มสุด) : {both:.1%}   <- เดามั่ว ~25%")
print(f"ตัวอักษรที่ตอบ         : {dict(letters)}")
print(f"เวลา: {time.time()-t0:.0f}s")

with open("results/eval_swap.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows_out[0].keys())
    w.writeheader(); w.writerows(rows_out)
print("บันทึกรายข้อไว้ที่ results/eval_swap.csv")
