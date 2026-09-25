r"""สะพานไปขั้นที่ 6: "อายุชีวภาพ" จากเลือด ทำนายการเสียชีวิตได้ไหม
1) GBoost ทายอายุจากเลือด (5-fold แบ่งตามคน) -> age gap = ทาย - จริง
2) ปรับ gap ด้วยอายุจริง (แก้ regression-to-the-mean)
3) Cox model: ความเสี่ยงเสียชีวิต ~ อายุจริง + เพศ + age gap"""
import json, re, time
import numpy as np, pandas as pd
from datasets import load_dataset
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from lifelines import CoxPHFitter

REPO = "insilicomedicine/longebench"
CONVERT = {
    ("Albumin", "g/L"): ("g/dL", 0.1), ("Creatinine", "mg/dL"): ("umol/L", 88.42),
    ("Serum glucose", "mmol/L"): ("mg/dL", 18.016), ("Blood urea nitrogen", "mmol/L"): ("mg/dL", 2.801),
    ("Total cholesterol", "mmol/L"): ("mg/dL", 38.67), ("HDL", "mmol/L"): ("mg/dL", 38.67),
    ("LDL", "mmol/L"): ("mg/dL", 38.67), ("Triglyceride", "mmol/L"): ("mg/dL", 88.57),
    ("Uric acid", "umol/L"): ("mg/dL", 1 / 59.48), ("Total bilirubin", "umol/L"): ("mg/dL", 1 / 17.1),
}
VAL = re.compile(r"([A-Za-z][A-Za-z0-9 \-]*?) \(([^)]+)\) (-?\d+(?:\.\d+)?)")
def parse(text):
    f = {}
    for name, unit, v in VAL.findall(text):
        name, v = name.strip(), float(v)
        if (name, unit) in CONVERT:
            v *= CONVERT[(name, unit)][1]
        f[name] = v
    m = re.search(r"BMI (\d+(?:\.\d+)?)", text)
    if m: f["BMI"] = float(m.group(1))
    f["male"] = 1.0 if re.search(r"\bmale\b|\bMale\b", text) and not re.search(r"\bfemale\b|\bFemale\b", text) else 0.0
    return f

t0 = time.time()
ds = load_dataset(REPO, "benchmark", streaming=True)
people, outcome = {}, {}
for row in ds[list(ds.keys())[0]]:
    t = row["task"]
    if t not in ("nhanes_age_regression", "nhanes_mortality_binary") or not row["metadata"]:
        continue
    fu = json.loads(row["metadata"])["follow_up"]
    user = [m for m in row["messages"] if m["role"] == "user"][-1]["content"]
    if t == "nhanes_age_regression":
        s, age = re.search(r"SEQN (\d+):\s*(\d+)", fu).groups()
        people[s] = (parse(user), int(age))
    else:
        s = re.search(r"SEQN (\d+)", fu).group(1)
        y = re.search(r"at (\d+) years?(?: (\d+) months?)?", fu)
        yrs = int(y.group(1)) + (int(y.group(2)) if y.group(2) else 0) / 12
        age_m = re.search(r"(\d+)-year-old", user)
        outcome[s] = dict(dead=int("Deceased" in fu), time=yrs,
                          age_mort=int(age_m.group(1)) if age_m else np.nan,
                          male=1 if re.search(r"-year-old male", user) else 0)
common = sorted(set(people) & set(outcome))
print(f"โหลด {len(people)} คน (อายุ) / {len(outcome)} คน (ติดตาม) / ตรงกัน {len(common)} ({time.time()-t0:.0f}s)")

# ตรวจความสอดคล้อง: อายุจริงจากสองชุดต้องตรงกัน
ages = np.array([people[s][1] for s in common]); ages_m = np.array([outcome[s]["age_mort"] for s in common])
ok = ~np.isnan(ages_m)
print(f"ตรวจ: อายุจริงสองชุดตรงกัน {np.mean(ages[ok] == ages_m[ok]):.1%} | ต่างกันเฉลี่ย {np.mean(np.abs(ages[ok]-ages_m[ok])):.2f} ปี")

cols = sorted({k for s in common for k in people[s][0]})
X = np.array([[people[s][0].get(c, np.nan) for c in cols] for s in common])
y = ages.astype(float)
oof = np.zeros(len(y))
for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
    oof[te] = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=0).fit(X[tr], y[tr]).predict(X[te])
print(f"GBoost MAE {np.abs(oof-y).mean():.1f} ปี")

gap = oof - y
slope, icpt = np.polyfit(y, gap, 1)
gap_adj = gap - (slope * y + icpt)       # age gap ที่ไม่ขึ้นกับอายุจริง
print(f"ก่อนปรับ: gap กับอายุจริง correlation = {np.corrcoef(y, gap)[0,1]:.2f} -> หลังปรับ {np.corrcoef(y, gap_adj)[0,1]:.2f}")

df = pd.DataFrame({
    "time": [outcome[s]["time"] for s in common],
    "dead": [outcome[s]["dead"] for s in common],
    "age10": y / 10,                     # ต่อ 10 ปี
    "male": [outcome[s]["male"] for s in common],
    "gap5": gap_adj / 5,                 # ต่อ 5 ปี
})
print(f"\nเสียชีวิต {df.dead.sum()} / {len(df)} คน | ติดตามเฉลี่ย {df.time.mean():.1f} ปี")

base = CoxPHFitter().fit(df[["time", "dead", "age10", "male"]], "time", "dead")
full = CoxPHFitter().fit(df, "time", "dead")
r = full.summary.loc["gap5"]
print("\n===== ผล Cox model =====")
print(f"เลือดแก่กว่าอายุจริงทุก 5 ปี -> ความเสี่ยงเสียชีวิต x{r['exp(coef)']:.2f} "
      f"[95% CI {r['exp(coef) lower 95%']:.2f}–{r['exp(coef) upper 95%']:.2f}], p = {r['p']:.2g}")
ra = full.summary.loc["age10"]
print(f"(เทียบ: อายุจริงมากขึ้นทุก 10 ปี -> x{ra['exp(coef)']:.2f})")
print(f"C-index: อายุ+เพศ = {base.concordance_index_:.3f} -> +age gap = {full.concordance_index_:.3f}  (สูงขึ้น = ทำนายดีขึ้น)")

# ภาพง่ายๆ: แบ่ง 3 กลุ่มตาม gap, ดูอัตราตายใน 10 ปี (เฉพาะคนที่ติดตามได้ครบ 10 ปี หรือตายก่อน)
df["group"] = pd.qcut(gap_adj, 3, labels=["เลือดอ่อนกว่าอายุ", "กลางๆ", "เลือดแก่กว่าอายุ"])
known = df[(df.time >= 10) | (df.dead == 1)]
print("\nอัตราเสียชีวิตภายใน 10 ปี (แยกตามกลุ่ม, คนอายุใกล้เคียงกันถูกกระจายทุกกลุ่มแล้ว):")
for g, sub in known.groupby("group", observed=True):
    d10 = ((sub.dead == 1) & (sub.time < 10)).mean()
    print(f"  {g:22} {d10:6.1%}  (n={len(sub)}, อายุเฉลี่ย {sub.age10.mean()*10:.0f})")
