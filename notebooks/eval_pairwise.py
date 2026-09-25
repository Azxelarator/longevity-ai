"""Mk.1: รัน LFM2-1.2B-Longevity บนเครื่อง แล้ววัดคะแนนเองบน nhanes_age_pairwise (45 ข้อ)
เฉลยคำนวณจาก metadata.follow_up (คนที่อายุมากกว่า = คำตอบที่ถูก)"""
import json, re, time
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "LiquidAI/LFM2-1.2B-Longevity"
TASK = "nhanes_age_pairwise"
LIMIT = 45  # ลดเหลือ 5 ถ้าอยากลองเร็วๆ

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16 if device == "cuda" else torch.float32
).to(device).eval()

def gold_answer(meta):
    ages = re.findall(r"Participant ([AB])[^:]*:\s*(\d+)", json.loads(meta)["follow_up"])
    d = {k: int(v) for k, v in ages}
    return "A" if d["A"] > d["B"] else "B"

ds = load_dataset("insilicomedicine/longebench", "mini", streaming=True)["eval"]
correct = total = 0
t0 = time.time()
for row in ds:
    if row["task"] != TASK:
        continue
    gold = gold_answer(row["metadata"])
    enc = tok.apply_chat_template(
        row["messages"], add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=5, do_sample=False)
    reply = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    pred = next((c for c in reply.upper() if c in "AB"), "?")
    total += 1
    correct += pred == gold
    print(f"{total:3}  gold={gold} pred={pred}  {'✓' if pred == gold else '✗'}  raw={reply!r}")
    if total >= LIMIT:
        break

print(f"\nAccuracy: {correct}/{total} = {correct/total:.1%}   (เดามั่ว = 50%)")
print(f"เวลา: {time.time()-t0:.1f} วินาที")
if device == "cuda":
    print(f"VRAM สูงสุด: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
