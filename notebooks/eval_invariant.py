r"""Mk.4: ทำให้คำตอบไม่ขึ้นกับตำแหน่ง "โดยการออกแบบ" (ขั้นที่ 2 ของ ROADMAP)
แทนที่จะให้โมเดลพิมพ์ A/B เราอ่าน "ความน่าจะเป็น" ของ A กับ B โดยตรง แล้วถาม 2 ลำดับ:
  ลำดับปกติ   : P(ข้อมูล X แก่กว่า) = P(A)
  ลำดับสลับก้อน: P(ข้อมูล X แก่กว่า) = P(B)   (X ย้ายไปอยู่ตำแหน่ง B)
คะแนนสุดท้าย = ค่าเฉลี่ยของสองค่า -> สลับยังไงก็ได้คำตอบเดิม (consistency 100% โดยนิยาม)
ใช้:  run notebooks\eval_invariant.py [จำนวนข้อ]   (ค่าเริ่มต้น 200)"""
import csv, json, re, sys, time, warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "LiquidAI/LFM2-1.2B-Longevity"
TASK = "nhanes_age_pairwise"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 200

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()

# token id ของคำตอบ "A" และ "B"
def tid(s):
    ids = tok.encode(s, add_special_tokens=False)
    assert len(ids) == 1, (s, ids)
    return ids[0]
ID_A, ID_B = tid("A"), tid("B")

def gold_answer(meta):
    d = dict(re.findall(r"Participant ([AB])[^:]*:\s*(\d+)", json.loads(meta)["follow_up"]))
    return ("A" if int(d["A"]) > int(d["B"]) else "B"), d

def block_swap(text):
    pre, rest = text.split("Participant A:", 1)
    a, b = rest.split("Participant B:", 1)
    return f"{pre}Participant A:{b.rstrip()}\n\nParticipant B:{a.rstrip()}"

def p_a(messages):
    """ความน่าจะเป็นที่โมเดลจะตอบ A (เทียบกับ B เท่านั้น)"""
    ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"].to(device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, -1]
    two = torch.stack([logits[ID_A], logits[ID_B]]).float().softmax(0)
    return two[0].item()

ds = load_dataset("insilicomedicine/longebench", "benchmark", streaming=True)
split = list(ds.keys())[0]
rows, t0 = [], time.time()
for row in ds[split]:
    if row["task"] != TASK:
        continue
    gold, ages = gold_answer(row["metadata"])
    msgs = row["messages"]
    ui = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    sw = [dict(m) for m in msgs]
    sw[ui]["content"] = block_swap(msgs[ui]["content"])
    pa1 = p_a(msgs)          # P(ข้อมูลเดิมของ A แก่กว่า) จากลำดับปกติ
    pa2 = 1 - p_a(sw)        # ข้อมูลเดิมของ A อยู่ตำแหน่ง B ในลำดับสลับ -> P(B)
    score = (pa1 + pa2) / 2  # คะแนนที่ไม่ขึ้นกับตำแหน่ง
    rows.append(dict(i=len(rows) + 1, age_A=ages["A"], age_B=ages["B"], gold=gold,
                     pA_orig=round(pa1, 4), pA_swapped=round(pa2, 4), score=round(score, 4),
                     pred_single="A" if pa1 > 0.5 else "B",
                     pred_invariant="A" if score > 0.5 else "B"))
    if len(rows) % 50 == 0:
        print(f"{len(rows)} ข้อ... ({time.time()-t0:.0f}s)")
    if len(rows) >= LIMIT:
        break

n = len(rows)
acc_single = sum(r["pred_single"] == r["gold"] for r in rows) / n
acc_inv = sum(r["pred_invariant"] == r["gold"] for r in rows) / n
predA = sum(r["pred_invariant"] == "A" for r in rows)
bias_raw = sum(r["pA_orig"] for r in rows) / n
print(f"\n===== Mk.4 ({n} ข้อ, {TASK}) =====")
print(f"ถามลำดับเดียว (argmax)      : {acc_single:.1%}")
print(f"order-invariant (เฉลี่ย 2 ลำดับ): {acc_inv:.1%}   consistency = 100% โดยนิยาม")
print(f"ค่าเฉลี่ย P(A) ลำดับเดียว    : {bias_raw:.2f}  (ไม่เอียง = 0.50)")
print(f"ทาย A/B หลังแก้              : {predA}/{n-predA}")

gap = lambda r: abs(int(r["age_A"]) - int(r["age_B"]))
print("\nตามช่วงอายุห่าง (order-invariant):")
for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 200)]:
    b = [r for r in rows if lo <= gap(r) < hi]
    if b:
        print(f"  {lo:>2}-{hi:<3} ปี: {sum(r['pred_invariant']==r['gold'] for r in b)/len(b):.1%} ({len(b)} ข้อ)")
print(f"เวลา: {time.time()-t0:.0f}s")

with open("results/eval_invariant.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("บันทึกไว้ที่ results/eval_invariant.csv")
