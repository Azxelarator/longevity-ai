r"""Mk.3: ทดสอบสลับป้าย A<->B ทุก domain ที่เป็น pairwise (อายุ)
ใช้:  run notebooks\eval_swap_all.py check   -> โหมดตรวจ: โชว์การสลับ ไม่รันโมเดล
      run notebooks\eval_swap_all.py         -> รันจริง"""
import csv, json, re, sys, time, warnings
warnings.filterwarnings("ignore", category=UserWarning)
from collections import Counter, defaultdict
from datasets import load_dataset

CHECK = len(sys.argv) > 1 and sys.argv[1] == "check"
LIMIT_PER_TASK = 30
TASKS = ["nhanes_age_pairwise", "geodnam_age_pairwise", "gtex_age_pairwise", "olink_age_pairwise"]

# ---------- เฉลย: คืน "A" หรือ "B" (ฝ่ายที่แก่กว่า) หรือ None ถ้าเท่ากัน ----------
def first_num(s):
    return int(re.search(r"\d+", s).group())

def gold_of(task, meta):
    fu = json.loads(meta)["follow_up"]
    if task == "nhanes_age_pairwise":
        d = dict(re.findall(r"Participant ([AB])[^:]*:\s*(\d+)", fu))
        a, b = int(d["A"]), int(d["B"])
    elif task == "geodnam_age_pairwise":
        d = dict(re.findall(r"Patient-([AB]):\s*(\d+)", fu))
        a, b = int(d["A"]), int(d["B"])
    elif task == "gtex_age_pairwise":
        d = dict(re.findall(r"Sample-([AB])\s*\([^)]*\):\s*(\d+)", fu))
        a, b = int(d["A"]), int(d["B"])
    elif task == "olink_age_pairwise":
        d = dict(re.findall(r"Patient-([AB]):\s*[^,]*,\s*(\d+)", fu))
        a, b = int(d["A"]), int(d["B"])
    if a == b:
        return None
    return "A" if a > b else "B"

# ---------- ตัวสลับป้าย ----------
FLIP = {"A": "B", "B": "A", "a": "b", "b": "a"}
LABEL_RE = re.compile(
    r"\b(?P<pre>Participant |Sample |Sample-|Patient-)(?P<l1>[ABab])\b"   # ป้ายมีคำนำหน้า
    r"|\b(?P<l2>[AB])(?=: )"                                              # A: 0.73 / A: A male
    r"|\b(?P<l3>[AB])(?= (?:much )?(?:higher|lower))"                     # A higher / B much higher
)
OPT_SIMPLE = re.compile(r"Options: A\. [^\n]*? B\. [^\n]*")

def swap_labels(text):
    def rep(m):
        if m.group("l1"):
            return m.group("pre") + FLIP[m.group("l1")]
        l = m.group("l2") or m.group("l3")
        return FLIP[l]
    orig_opts = OPT_SIMPLE.findall(text)
    out = LABEL_RE.sub(rep, text)
    # บรรทัด Options แบบ "A. Sample A  B. Sample B" ต้องคงเดิม (ตัวเลือก A = ป้าย A เสมอ)
    new_opts = OPT_SIMPLE.findall(out)
    for o, n in zip(orig_opts, new_opts):
        out = out.replace(n, o, 1)
    return out

def label_counts(text):
    c = Counter()
    for m in LABEL_RE.finditer(text):
        c[(m.group("l1") or m.group("l2") or m.group("l3")).upper()] += 1
    return c

# ---------- โหลดข้อสอบ ----------
ds = load_dataset("insilicomedicine/longebench", "benchmark", streaming=True)
split = list(ds.keys())[0]
items = defaultdict(list)
for row in ds[split]:
    t = row["task"]
    if t in TASKS and len(items[t]) < LIMIT_PER_TASK:
        g = gold_of(t, row["metadata"])
        if g:
            items[t].append((row["messages"], g))
    if all(len(items[t]) >= LIMIT_PER_TASK for t in TASKS):
        break

if CHECK:
    for t in TASKS:
        msgs, g = items[t][0]
        u = [m for m in msgs if m["role"] == "user"][-1]["content"]
        s = swap_labels(u)
        print("=" * 90)
        print(f"{t}  | ได้ {len(items[t])} ข้อ | เฉลยข้อแรก={g} -> หลังสลับ={FLIP[g]}")
        print("นับป้ายก่อนสลับ:", dict(label_counts(u)), " หลังสลับ:", dict(label_counts(s)))
        print("--- ก่อน (ต้น 350) ---\n" + u[:350])
        print("--- หลัง (ต้น 350) ---\n" + s[:350])
        print("--- ก่อน (ท้าย 250) ---\n" + u[-250:])
        print("--- หลัง (ท้าย 250) ---\n" + s[-250:])
    sys.exit(0)

# ---------- รันโมเดลจริง ----------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL = "LiquidAI/LFM2-1.2B-Longevity"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()

from torch.nn.attention import sdpa_kernel, SDPBackend
import transformers.integrations.sdpa_attention as _sa
# โมเดลใช้ GQA (query 32 หัว, key/value 8 หัว) แต่ kernel ประหยัดหน่วยความจำต้องการจำนวนหัวเท่ากัน
# -> ปิดโหมด GQA ใน SDPA ให้ transformers ขยาย key/value เป็น 32 หัวก่อน
_sa.use_gqa_in_sdpa = lambda *a, **k: False
EFFICIENT = [SDPBackend.EFFICIENT_ATTENTION]   # Windows build ไม่มี flash attention

def ask(messages):
    """คืน (คำตอบ, จำนวนโทเคน) — ถ้า VRAM ไม่พอ คืน ("SKIP", จำนวนโทเคน)"""
    ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)["input_ids"].to(device)
    n = ids.shape[1]
    try:
        # ส่งแค่ input_ids (ไม่มี mask) + บังคับ attention แบบประหยัดหน่วยความจำ
        with torch.no_grad(), sdpa_kernel(EFFICIENT):
            out = model.generate(input_ids=ids, max_new_tokens=5, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
    except (torch.OutOfMemoryError, RuntimeError) as e:
        torch.cuda.empty_cache()
        print(f"  ! {type(e).__name__}: {str(e)[:120]}")
        return "SKIP", n
    reply = tok.decode(out[0, n:], skip_special_tokens=True).strip()
    return next((c for c in reply.upper() if c in "AB"), "?"), n

rows = []
skipped = Counter()
for t in TASKS:
    t0 = time.time()
    for i, (msgs, g) in enumerate(items[t]):
        ui = max(k for k, m in enumerate(msgs) if m["role"] == "user")
        sw = [dict(m) for m in msgs]
        sw[ui]["content"] = swap_labels(msgs[ui]["content"])
        p1, ntok = ask(msgs)
        p2, _ = ask(sw) if p1 != "SKIP" else ("SKIP", ntok)
        if "SKIP" in (p1, p2):
            skipped[t] += 1
            print(f"  ข้าม {t} ข้อ {i+1}: {ntok} โทเคน (VRAM ไม่พอ)")
            continue
        rows.append(dict(task=t, i=i + 1, tokens=ntok, gold=g, pred=p1,
                         gold_swapped=FLIP[g], pred_swapped=p2,
                         consistent=int(p1 != p2 and "?" not in (p1, p2))))
    print(f"{t}: เสร็จ {len(items[t])} ข้อ, ข้าม {skipped[t]} ({time.time()-t0:.0f}s)")
    if device == "cuda":
        print(f"  VRAM สูงสุด: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

print(f"\n{'task':24} {'n':>3} {'acc':>6} {'acc_sw':>7} {'consist':>8} {'both':>6} {'sure_acc':>9} {'A/B':>9} {'tok':>6}")
for t in TASKS:
    r = [x for x in rows if x["task"] == t]
    n = len(r)
    if n == 0:
        print(f"{t[:24]:24}   0  (ข้ามหมด)"); continue
    acc = sum(x["pred"] == x["gold"] for x in r) / n
    acc2 = sum(x["pred_swapped"] == x["gold_swapped"] for x in r) / n
    cons = sum(x["consistent"] for x in r) / n
    both = sum(x["pred"] == x["gold"] and x["pred_swapped"] == x["gold_swapped"] for x in r) / n
    sure = [x for x in r if x["consistent"]]
    sacc = (sum(x["pred"] == x["gold"] for x in sure) / len(sure)) if sure else float("nan")
    lc = Counter([x["pred"] for x in r] + [x["pred_swapped"] for x in r])
    tk = sum(x["tokens"] for x in r) // n
    print(f"{t[:24]:24} {n:>3} {acc:>6.0%} {acc2:>7.0%} {cons:>8.0%} {both:>6.0%} {sacc:>9.0%} {lc['A']:>4}/{lc['B']:<4} {tk:>6}")
print("\nboth = ถูกทั้งสองรอบ (สุ่ม ~25%) | sure_acc = ความแม่นเฉพาะข้อที่ consistent | tok = ความยาว prompt เฉลี่ย")

with open("results/eval_swap_all.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
print("บันทึกไว้ที่ results/eval_swap_all.csv")
