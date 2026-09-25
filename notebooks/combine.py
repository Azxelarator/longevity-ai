r"""ลองรวม GBoost + LLM บน nhanes_age_pairwise (ไม่ใช้ GPU)
ต้องมี: results/eval_invariant_all.csv (จาก eval_invariant.py all)
       results/baseline_pairs.csv     (จาก baseline.py)"""
import csv, math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

llm = list(csv.DictReader(open("results/eval_invariant_all.csv", encoding="utf-8")))
gb = list(csv.DictReader(open("results/baseline_pairs.csv", encoding="utf-8")))
assert len(llm) == len(gb), (len(llm), len(gb))
bad = sum((a["age_A"], a["age_B"]) != (b["age_A"], b["age_B"]) for a, b in zip(llm, gb))
print(f"จับคู่ {len(llm)} ข้อ | อายุไม่ตรงกัน {bad} ข้อ (ต้องเป็น 0)")
assert bad == 0

y = np.array([1 if r["gold"] == "A" else 0 for r in llm])
p = np.clip(np.array([float(r["score"]) for r in llm]), 1e-4, 1 - 1e-4)
llm_logit = np.log(p / (1 - p))                                   # ความมั่นใจของ LLM ว่า A แก่กว่า
gb_diff = np.array([float(r["predA"]) - float(r["predB"]) for r in gb])  # อายุที่ GBoost ทาย: A - B (ปี)

llm_ok = (llm_logit > 0) == (y == 1)
gb_ok = (gb_diff > 0) == (y == 1)
n = len(y)
print(f"\nLLM ถูก {llm_ok.mean():.1%} | GBoost ถูก {gb_ok.mean():.1%}")
print("ตารางถูก/ผิด:")
print(f"  ถูกทั้งคู่            : {( llm_ok &  gb_ok).sum():5}")
print(f"  GBoost ถูก, LLM ผิด  : {(~llm_ok &  gb_ok).sum():5}")
print(f"  LLM ถูก, GBoost ผิด  : {( llm_ok & ~gb_ok).sum():5}   <- ข้อที่ LLM ช่วยได้")
print(f"  ผิดทั้งคู่            : {(~llm_ok & ~gb_ok).sum():5}")
print(f"  ความสัมพันธ์ของสองตัว (correlation): {np.corrcoef(llm_logit, gb_diff)[0,1]:.2f}")

# รวมด้วย logistic regression + 5-fold CV (ไม่ใช้ข้อที่ทายในการฝึก)
X = np.column_stack([gb_diff, llm_logit])
oof = np.zeros(n)
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
    m = LogisticRegression().fit(X[tr], y[tr])
    oof[te] = m.predict_proba(X[te])[:, 1]
comb_ok = (oof > 0.5) == (y == 1)

def wilson(k, n, z=1.96):
    q = k / n; d = 1 + z*z/n
    c = (q + z*z/(2*n)) / d; h = z * ((q*(1-q)/n + z*z/(4*n*n)) ** 0.5) / d
    return c - h, c + h
for name, ok in [("LLM", llm_ok), ("GBoost", gb_ok), ("รวมกัน", comb_ok)]:
    lo, hi = wilson(ok.sum(), n)
    print(f"{name:8}: {ok.mean():.1%}  [95% CI {lo:.1%}, {hi:.1%}]")

# ทดสอบว่า "รวมกัน" ดีกว่า GBoost จริงไหม (McNemar: ดูเฉพาะข้อที่สองวิธีตอบต่างกัน)
b = int((comb_ok & ~gb_ok).sum()); c = int((~comb_ok & gb_ok).sum())
z = (abs(b - c) - 1) / math.sqrt(b + c) if b + c else 0
pval = math.erfc(z / math.sqrt(2))
print(f"\nรวมกัน vs GBoost: รวมถูกแต่ GB ผิด {b} ข้อ | GB ถูกแต่รวมผิด {c} ข้อ | p = {pval:.3f}  (< 0.05 = ต่างจริง)")
m = LogisticRegression().fit(X, y)
print(f"น้ำหนัก: GBoost {m.coef_[0][0]:.3f}/ปี, LLM {m.coef_[0][1]:.3f}/logit")
