r"""(ส่วนโหลด คัดลอกจาก mortality.py) สะพานไปขั้นที่ 6: "อายุชีวภาพ" จากเลือด ทำนายการเสียชีวิตได้ไหม
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


# ---------- PhenoAge (Levine et al., Aging 2018) ----------
# ใช้หน่วยตามเปเปอร์: albumin g/L, creatinine umol/L, glucose mmol/L, CRP mg/dL (ln), lymph %, MCV fL, RDW %, ALP U/L, WBC 1000/uL
NEED = ["Albumin", "Creatinine", "Serum glucose", "C-reactive protein", "Lymphocyte percent",
        "Mean cell volume", "Red cell distribution width", "Alkaline phosphatase", "White blood cell count"]
def phenoage(f, age):
    if any(k not in f for k in NEED):
        return np.nan
    xb = (-19.907 - 0.0336 * f["Albumin"] * 10 + 0.0095 * f["Creatinine"]
          + 0.1953 * f["Serum glucose"] / 18.016 + 0.0954 * np.log(max(f["C-reactive protein"], 0.01))
          - 0.0120 * f["Lymphocyte percent"] + 0.0268 * f["Mean cell volume"]
          + 0.3306 * f["Red cell distribution width"] + 0.00188 * f["Alkaline phosphatase"]
          + 0.0554 * f["White blood cell count"] + 0.0804 * age)
    g = 0.0076927
    M = 1 - np.exp(-np.exp(xb) * (np.exp(120 * g) - 1) / g)
    return 141.50225 + np.log(-0.00553 * np.log(1 - M)) / 0.090165

pa = np.array([phenoage(people[s][0], people[s][1]) for s in common])
have = ~np.isnan(pa)
print(f"\nPhenoAge คำนวณได้ {have.sum()} / {len(pa)} คน (ที่เหลือขาดค่าเลือดบางตัว)")
print(f"ตรวจ: PhenoAge กับอายุจริง correlation = {np.corrcoef(pa[have], y[have])[0,1]:.2f} (ควรสูง ~0.9), "
      f"ต่างกันเฉลี่ย {np.mean(pa[have]-y[have]):+.1f} ปี")

def adj(g, a):
    sl, ic = np.polyfit(a, g, 1)
    return g - (sl * a + ic)

yy = y[have]
d = pd.DataFrame({
    "time": [outcome[s]["time"] for s, h in zip(common, have) if h],
    "dead": [outcome[s]["dead"] for s, h in zip(common, have) if h],
    "age10": yy / 10,
    "male": [outcome[s]["male"] for s, h in zip(common, have) if h],
    "gb_gap5": adj(oof[have] - yy, yy) / 5,
    "pa_gap5": adj(pa[have] - yy, yy) / 5,
})
print(f"คนที่ใช้เทียบ: {len(d)} | เสียชีวิต {d.dead.sum()}\n")

def cox(cols, label):
    m = CoxPHFitter().fit(d[["time", "dead"] + cols], "time", "dead")
    parts = []
    for c in cols:
        if c.endswith("gap5"):
            r = m.summary.loc[c]
            parts.append(f"{c}: x{r['exp(coef)']:.2f} [{r['exp(coef) lower 95%']:.2f}–{r['exp(coef) upper 95%']:.2f}] p={r['p']:.1g}")
    print(f"{label:28} C-index {m.concordance_index_:.3f}   " + " | ".join(parts))
    return m

print("===== เทียบตาชั่ง: เลือดแก่กว่าอายุทุก 5 ปี -> ความเสี่ยงเสียชีวิต =====")
cox(["age10", "male"], "อายุ + เพศ (ฐาน)")
cox(["age10", "male", "gb_gap5"], "+ GBoost ของเรา")
cox(["age10", "male", "pa_gap5"], "+ PhenoAge")
cox(["age10", "male", "gb_gap5", "pa_gap5"], "+ ทั้งสองตัว")

# ตารางแบบคุมอายุ: แบ่ง tertile ของ gap "ภายในช่วงอายุเดียวกัน"
d["band"] = pd.cut(d.age10 * 10, [0, 40, 60, 200], labels=["<40", "40-60", "60+"])
print("\nอัตราเสียชีวิตภายใน 10 ปี ตาม PhenoAge gap (แบ่ง 3 กลุ่มภายในแต่ละช่วงอายุ — คุมอายุแล้ว):")
known = d[(d.time >= 10) | (d.dead == 1)].copy()
known["tert"] = known.groupby("band", observed=True)["pa_gap5"].transform(lambda s: pd.qcut(s, 3, labels=[0, 1, 2]).astype(int))
for b, sub in known.groupby("band", observed=True):
    rates = [((g.dead == 1) & (g.time < 10)).mean() for _, g in sub.groupby("tert")]
    print(f"  อายุ {b:6}: อ่อนกว่าอายุ {rates[0]:5.1%} | กลางๆ {rates[1]:5.1%} | แก่กว่าอายุ {rates[2]:5.1%}  (n={len(sub)})")
