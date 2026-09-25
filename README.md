# Longevity AI — stress-testing aging-biology LLMs on a single laptop

> **Status:** early, independent, non-profit research. No guarantees of success — but every result, good or bad, is reported here.
> ภาษาไทย: [README.th.md](README.th.md)

## Why
The long-term goal is a tool that helps find ways to slow cellular aging.
Before we can test whether anything slows aging, we need a reliable **scale** — a model that reads biological data and tells us how old a body really is. This project starts by testing how reliable that scale is.

## What we tested
- **Model:** [`LiquidAI/LFM2-1.2B-Longevity`](https://huggingface.co/LiquidAI/LFM2-1.2B-Longevity)
- **Benchmark:** [LongevityBench](https://huggingface.co/datasets/insilicomedicine/longebench) (Zhavoronkov et al., *Cell* 2026, [doi:10.1016/j.cell.2026.08.026](https://doi.org/10.1016/j.cell.2026.08.026))
- **Hardware:** one laptop — RTX 4050 Laptop (6 GB VRAM), 16 GB RAM. No cloud, no budget.

## Findings so far
Task: `nhanes_age_pairwise` — given two people's blood test results, which one is older?

| # | Experiment | n | Result |
|---|---|---|---|
| Mk.1 | Plain run (mini split) | 45 | 60.0% accuracy — too few items to beat chance convincingly |
| Mk.2 | **Swap test**: ask again with A/B data blocks swapped | 200 | 69.5% normal / 77.5% swapped, but only **65% consistent**. Answered "B" 266 vs "A" 134 → strong **letter/position bias** |
| Mk.2.1 | Trust only answers that stay consistent after swapping | 200 | **86.2%** accuracy, but only answers 65% of items |
| Mk.3 | **Label swap** (rename A↔B, keep data in place) on 4 domains | 30 each | Accuracy collapses after relabeling: blood 70→47%, DNA methylation 80→33%, transcriptomics 67→40%, proteomics 47→33% → the model binds answers to **position, not label** |
| Mk.4 | **Order-invariant scoring**: read P(A) vs P(B), average over both orders | 200 | **79.0%** accuracy, **100% consistency by construction**, answers every item. Mean P(A) was 0.39 before the fix (unbiased = 0.50) |
| Mk.4 (full) | Same, on the **entire** task | 2,102 | Single-order **71.2%** [95% CI 69.2–73.1] → order-invariant **79.2%** [77.4–80.8]. Intervals do not overlap. |

### Stage 3: simple baselines beat the LLM
Same 2,102 pairs. Baselines are trained on each person's blood values from `nhanes_age_regression` with **5-fold cross-validation split by person**, so every age prediction comes from a model that never saw that person (all 4,204 pairwise participants also appear in the regression task — training on it directly would leak). Units are harmonized first (e.g. creatinine mg/dL → µmol/L).

| Method | Pairwise accuracy | 95% CI | Age MAE |
|---|---|---|---|
| LFM2-1.2B-Longevity, order-invariant (Mk.4) | 79.2% | 77.4–80.8 | — |
| Ridge regression on 30 blood/biometric values | 82.8% | 81.1–84.3 | 9.7 yrs |
| **Gradient boosting** (HistGradientBoosting) | **87.9%** | **86.5–89.2** | **7.7 yrs** |

The gradient-boosting baseline wins in every age-gap bin. **For predicting age from blood values alone, a few-second CPU model beats this 1.2B LLM.** In fairness: the baselines are fit on ~3,300 people from the same distribution, while the LLM is used zero-shot; and we have not yet measured the LLM's possible advantages (multiple data types, natural-language explanations).

Accuracy by age gap (Mk.4, full 2,102 items): 0–10 yrs 57.7% · 10–20 yrs 71.2% · 20–40 yrs 86.2% · 40+ yrs 96.0%.
**Remaining weakness:** distinguishing people less than ~20 years apart.

### Caveats
- LongevityBench releases only the `eval` split; we cannot independently check train/test overlap (the paper splits NHANES by survey wave).
- The dataset card does not report per-task scores, so we have not yet compared our numbers to the paper's.
- Mk.3 uses 30 items per domain (±~15% noise) and places "B" before "A", a format the model never saw in training — a harsher test than normal use.
- These are findings about robustness to prompt format, not a judgment of the original paper's results.

## Roadmap
See [ROADMAP.md](ROADMAP.md). Short version:
1. ✅ Better than chance
2. ✅ (blood only) Answers don't change when the question format changes
3. ❌ Beat a simple baseline — **not passed** on blood data: gradient boosting 87.9% vs LLM 79.2%
4. ⏳ Generalize to unseen data sources
5. Clinical use — requires doctors and clinical studies, not a solo project

## Want to help?
Open issues are the best place to start. Things nobody has done yet:
- Apply Mk.4 order-invariant scoring to the DNA methylation, transcriptomics and proteomics tasks (DNA prompts average ~22k tokens — needs more VRAM than 6 GB comfortably allows)
- Build the stage-3 baseline for `nhanes_age_pairwise`
- Handle the `nhanes_mortality_pairwise` task (answers involve censored survival)
- Test the larger `LFM2-2.6B-Longevity` / `L-Qwen` models

## Run it yourself (Windows)
```
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
run notebooks\eval_invariant.py
```
Scripts in `notebooks/` are numbered by the experiment log in [LOG.md](LOG.md) (Thai).

**Tip for ≤ 6 GB GPUs:** on Windows the default attention path may try to allocate tens of GB on long prompts. `notebooks/eval_swap_all.py` shows the workaround (disable GQA-in-SDPA and force the memory-efficient kernel).

## License
Code: MIT (see [LICENSE](LICENSE)).
The LongevityBench data is CC-BY-NC-4.0 and the models have their own licenses — follow the original terms when using them.
