# ATP Tennis Match Analysis (2010–2024)

A statistical data science project analyzing resilience in professional men's tennis using 15 years of ATP match data. Four hypothesis-driven regression models test how clutch performance, physical recovery, surface specialization, and fatigue interact to determine match outcomes.

**Author:** Akinwale Agesin  
**Data source:** [Jeff Sackmann's ATP Repository](https://github.com/JeffSackmann/tennis_atp)

---

## Research Questions

1. **Short-term resilience:** Does clutch performance (break-point save rate) predict match outcomes, controlling for player rank?
2. **Long-term resilience:** After a fatiguing match, do players who serve above their seasonal average win more often?
3. **Surface specialization:** Does playing on a preferred surface help players overcome ranking disadvantages?
4. **Fatigue & clutch interaction:** Do longer previous matches predict worse break-point performance?

---

## Project Structure

```
atp-tennis-analysis/
├── data_pipeline.ipynb        # Part 1: Data collection, cleaning, feature engineering, EDA
├── analysis.ipynb             # Part 2: Hypothesis testing, regression models, results
├── players_df.csv             # Player-match level dataset (output of data_pipeline.ipynb)
├── analysis_df.csv            # Regression-ready dataset (output of data_pipeline.ipynb)
├── requirements.txt
├── .gitignore
└── README.md
```

### Notebook Pipeline

```
data_pipeline.ipynb
      │
      │  produces
      ▼
players_df.csv ──────────┐
analysis_df.csv ─────────┴──▶  analysis.ipynb
                                (4 regression models,
                                 hypothesis testing,
                                 visualizations)
```

Run `data_pipeline.ipynb` first to generate the `.csv` files, then run `analysis.ipynb`.

---

## Hypotheses & Models

### H1 — Clutch Performance → Win Probability
**Model:** Logistic regression  
`logit(P(win)) = β₀ + β₁·clutch + β₂·rank`

**Finding:** Clutch is a strong, statistically significant predictor (β=0.954, p<0.001). A 0.1 increase in the clutch index raises the odds of winning by ~10%.

---

### H2 — Recovery Serving Performance → Next Match Win
**Model:** Logistic regression on fatigue-flagged matches (top 25% by duration)  
`logit(P(next_win)) = β₀ + β₁·next_FS_pct_diff`

**Finding:** Serving above one's seasonal average after a fatiguing match is significantly associated with winning the recovery match (β=2.893, p<0.001). A 10pp serving improvement raises win odds by ~34%.

---

### H3 — Surface Specialization → Win Probability (with rank interaction)
**Model:** Logistic regression with interaction term  
`logit(P(win)) = β₀ + β₁·rank_diff + β₂·surface_experience + β₃·(rank_diff × surface_experience)`

**Finding:** Surface experience has a strong positive effect (β=0.221, p<0.001). The negative interaction term (β=-0.006) shows specialization does not amplify upsets — specialists win consistently regardless of ranking gap.

---

### H4 — Fatigue (Previous Match Duration) → Clutch Performance
**Model:** OLS regression with surface fixed effects  
`clutch = β₀ + β₁·prev_minutes + β₂·rank + β₃·C(surface)`

**Finding:** Null hypothesis not rejected. `prev_minutes` coefficient is near zero and non-significant (p=0.093). No evidence that physical fatigue from a prior match degrades clutch performance.

---

## Engineered Features

| Feature | Description |
|---|---|
| `clutch` | Break points saved / break points faced per match |
| `FS_pct` | First serves in / total first serves attempted |
| `FS_pct_avg` | Player's average `FS_pct` across a full season |
| `FS_pct_diff` | Match `FS_pct` minus seasonal average (above/below baseline) |
| `next_FS_pct_diff` | `FS_pct_diff` in the player's next match |
| `prev_minutes` | Duration of the player's immediately preceding match |
| `surface_experience` | Proportion of career matches played on the current surface |
| `rank_diff` | Loser rank minus winner rank (match-level ranking gap) |

---

## Results Summary

| Hypothesis | Predictor | Effect | Significant? |
|---|---|---|---|
| H1 | Clutch index | ↑ Win probability | ✅ Yes |
| H2 | Recovery serving above average | ↑ Next win probability | ✅ Yes |
| H3 | Surface experience | ↑ Win probability | ✅ Yes |
| H3 (interaction) | Surface × rank gap | Dampens rank effect | ✅ Yes |
| H4 | Previous match duration | No effect on clutch | ❌ No |

---

## Setup & Usage

```bash
pip install -r requirements.txt
jupyter notebook
```

Open and run `data_pipeline.ipynb` first, then `analysis.ipynb`.

No data download required — `data_pipeline.ipynb` pulls directly from Jeff Sackmann's public GitHub repository at runtime.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
statsmodels
scipy
jupyter
```

---

## Limitations

- Match statistics (duration, break points) are consistently available only from ~2010 onward
- Data is aggregate match-level only — no point-by-point context (score state, weather, crowd)
- Clutch index is noisy for matches with few break points (extreme values of 0 or 1 are common artifacts)
- Sequential match analysis (H2, H4) relies on within-year continuity

---

## Skills Demonstrated

- **End-to-end data pipeline** — raw sports archive → engineered feature store → statistical models
- **Feature engineering** — domain-meaningful metrics from raw event data
- **Statistical inference** — logistic and OLS regression with preregistered hypotheses, confidence intervals, AUC, and residual diagnostics
- **Interaction modeling** — testing whether the effect of one variable changes as a function of another
- **Responsible AI alignment** — model scope documentation, extrapolation boundaries, and data governance principles
