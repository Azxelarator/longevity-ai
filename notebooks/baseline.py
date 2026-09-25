r"""ขั้นที่ 3: baseline แบบง่าย (ไม่ใช้ LLM) บน nhanes_age_pairwise
- อ่านค่าเลือดจาก nhanes_age_regression (1 คน/ข้อ + อายุจริง), แปลงหน่วยให้ตรงกัน
- 5-fold cross-validation แบ่งตามคน -> ทุกคนได้อายุที่ทายจากสูตรที่ไม่เคยเห็นเขา
- เอาอายุที่ทายไปตอบข้อสอบเทียบสองคน 2102 ข้อ แล้วเทียบกับ LLM (79.2%)"""
import csv, json, re, time
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

REPO = "insilicomedicine/longebench"

# ---------- แปลงหน่วยให้เป็นหน่วยเดียว ----------
# (ชื่อ, หน่วยที่เจอ) -> (หน่วยมาตรฐาน, ตัวคูณ)
CONVERT = {
    ("Albumin", "g/L"): ("g/dL", 0.1),
    ("Creatinine", "mg/dL"): ("umol/L", 88.42),
    ("Serum glucose", "mmol/L"): ("mg/dL", 18.016),
    ("Blood urea nitrogen", "mmol/L"): ("mg/dL", 2.801),
    ("Total cholesterol", "mmol/L"): ("mg/dL", 38.67),
    ("HDL", "mmol/L"): ("mg/dL", 38.67),
    ("LDL", "mmol/L"): ("mg/dL", 38.67),
    ("Triglyceride", "mmol/L"): ("mg/dL", 88.57),
    ("Uric acid", "umol/L"): ("mg/dL", 1 / 59.48),
    ("Total bilirubin", "umol/L"): ("mg/dL", 1 / 17.1),
}
VAL = re.compile(r"([A-Za-z][A-Za-z0-9 \-]*?) \(([^)]+)\) (-?\d+(?:\.\d+)?)")
units_seen = {}

def parse(text):
    f = {}
    for name, unit, v in VAL.findall(text):
        name = name.strip()
        v = float(v)
        if (name, unit) in CONVERT:
            unit, k = CONVERT[(name, unit)]
            v *= k
        units_seen.setdefault(name, set()).add(unit)
        f[name] = v
    m = re.search(r"BMI (\d+(?:\.\d+)?)", text)
    if m:
        f["BMI"] = float(m.group(1))
    f["male"] = 1.0 if re.search(r"\bMale\b|\bmale\b", text) else 0.0
    return f

# ---------- โหลด ----------
t0 = time.time()
ds = load_dataset(REPO, "benchmark", streaming=True)
split = list(ds.keys())[0]
people, pairs = {}, []
for row in ds[split]:
    t = row["task"]
    if t not in ("nhanes_age_regression", "nhanes_age_pairwise") or not row["metadata"]:
        continue
    fu = json.loads(row["metadata"])["follow_up"]
    if t == "nhanes_age_regression":
        seqn, age = re.search(r"SEQN (\d+):\s*(\d+)", fu).groups()
        user = [m for m in row["messages"] if m["role"] == "user"][-1]["content"]
        people[seqn] = (parse(user), int(age))
    else:
        d = re.findall(r"Participant ([AB]) \(SEQN (\d+)\):\s*(\d+)", fu)
        pairs.append({l: (s, int(a)) for l, s, a in d})
print(f"โหลด: {len(people)} คน, {len(pairs)} คู่ ({time.time()-t0:.0f}s)")

# ตรวจหน่วย: ถ้าชื่อไหนยังมีหลายหน่วย = ยังแปลงไม่ครบ
mixed = {k: v for k, v in units_seen.items() if len(v) > 1}
print("หน่วยที่ยังปนกัน (ต้องว่าง):", mixed if mixed else "ไม่มี ✓")

# ---------- ตาราง ----------
seqns = sorted(people)
cols = sorted({k for s in seqns for k in people[s][0]})
X = np.array([[people[s][0].get(c, np.nan) for c in cols] for s in seqns])
y = np.array([people[s][1] for s in seqns], dtype=float)
print(f"ตาราง: {X.shape[0]} คน x {X.shape[1]} ค่า | ค่าว่าง {np.isnan(X).mean():.0%}")

models = {
    "Ridge (เส้นตรง)": lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                             StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
    "GBoost (ต้นไม้)": lambda: HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0),
}

def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z * ((p*(1-p)/n + z*z/(4*n*n)) ** 0.5) / d
    return c - h, c + h

idx = {s: i for i, s in enumerate(seqns)}
missing = [p for p in pairs if p["A"][0] not in idx or p["B"][0] not in idx]
pairs = [p for p in pairs if p not in missing]
print(f"คู่ที่ใช้ได้: {len(pairs)} (หาคนไม่เจอ {len(missing)})\n")

for name, make in models.items():
    oof = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
        m = make().fit(X[tr], y[tr])
        oof[te] = m.predict(X[te])
    mae = np.abs(oof - y).mean()
    correct, bins = 0, {}
    for p in pairs:
        (sa, aa), (sb, ab) = p["A"], p["B"]
        gold = "A" if aa > ab else "B"
        pred = "A" if oof[idx[sa]] > oof[idx[sb]] else "B"
        ok = pred == gold
        correct += ok
        g = abs(aa - ab)
        key = "0-10" if g < 10 else "10-20" if g < 20 else "20-40" if g < 40 else "40+"
        bins.setdefault(key, []).append(ok)
    if name.startswith("GBoost"):  # เก็บผลรายคู่ไว้ให้ combine.py ใช้
        with open("results/baseline_pairs.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["i", "age_A", "age_B", "predA", "predB"])
            for i, p in enumerate(pairs, 1):
                w.writerow([i, p["A"][1], p["B"][1], round(oof[idx[p["A"][0]]], 3), round(oof[idx[p["B"][0]]], 3)])
    n = len(pairs); lo, hi = wilson(correct, n)
    print(f"== {name} ==")
    print(f"  ทายอายุคลาดเฉลี่ย (MAE): {mae:.1f} ปี")
    print(f"  ข้อสอบเทียบสองคน: {correct/n:.1%}  [95% CI {lo:.1%}, {hi:.1%}]")
    print("  ตามช่วงอายุห่าง: " + " | ".join(f"{k} {np.mean(bins[k]):.1%}" for k in ["0-10", "10-20", "20-40", "40+"] if k in bins))
print("\nเทียบ LLM (Mk.4 order-invariant): 79.2%  [77.4%, 80.8%]")
