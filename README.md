# ATP Tennis Match Analysis (2010–2024)

A data science project analyzing resilience in professional men's tennis using 15 years of ATP match data. I built four regression models to test how clutch performance, physical recovery, surface specialization, and fatigue interact to predict match outcomes.

**Author:** Akinwale Agesin  
**Data source:** [Jeff Sackmann's ATP Repository](https://github.com/JeffSackmann/tennis_atp)

---

## Research Questions

1. Does clutch performance (break-point save rate) actually predict who wins, once you control for ranking?
2. After a physically draining match, do players who serve better than their usual baseline win more in their next match?
3. Does playing on your best surface help you beat higher-ranked opponents?
4. Does a long previous match make players worse under pressure in their next one?

---

## Project Structure

```
atp-tennis-analysis/
├── data_pipeline.ipynb        # Data collection, cleaning, feature engineering, EDA
├── analysis.ipynb             # Hypothesis testing, regression models, results
├── players_df.csv             # Player-match level dataset (output of data_pipeline.ipynb)
├── analysis_df.csv            # Regression-ready dataset (output of data_pipeline.ipynb)
├── requirements.txt
├── .gitignore
└── README.md
```

Run `data_pipeline.ipynb` first to generate the CSVs, then open `analysis.ipynb`.

---

## Models & Findings

### H1 — Does clutch performance predict wins?
**Model:** Logistic regression — `logit(P(win)) = β₀ + β₁·clutch + β₂·rank`

Yes, pretty clearly. The clutch coefficient came out at 0.954 (p<0.001), meaning a 0.1 increase in the clutch index raises your odds of winning by about 10%. Rank still matters, but clutch adds real predictive power on top of it.

---

### H2 — Does serving well after a tough match predict winning the next one?
**Model:** Logistic regression on the top 25% of matches by duration  
`logit(P(next_win)) = β₀ + β₁·next_FS_pct_diff`

Also yes. Serving above your seasonal average after a fatiguing match is strongly associated with winning that recovery match (β=2.893, p<0.001). A 10 percentage point serving improvement roughly translates to 34% better odds of winning.

---

### H3 — Does surface specialization help you pull upsets?
**Model:** Logistic regression with interaction  
`logit(P(win)) = β₀ + β₁·rank_diff + β₂·surface_experience + β₃·(rank_diff × surface_experience)`

Surface experience has a strong positive effect (β=0.221, p<0.001), but the interaction term (β=-0.006) tells an interesting story — specialists don't actually win more upsets than expected. They just win consistently across all matchups, regardless of the ranking gap.

---

### H4 — Does a long previous match hurt clutch performance?
**Model:** OLS with surface fixed effects  
`clutch = β₀ + β₁·prev_minutes + β₂·rank + β₃·C(surface)`

No evidence of this. The `prev_minutes` coefficient is basically zero and not significant (p=0.093). Physical fatigue from the previous match doesn't seem to affect break-point performance, at least not in a way this model can detect.

---

## Features I Engineered

| Feature | What it captures |
|---|---|
| `clutch` | Break points saved / break points faced per match |
| `FS_pct` | First serves in / total first serves attempted |
| `FS_pct_avg` | Player's average first serve % across a full season |
| `FS_pct_diff` | Match first serve % minus seasonal average |
| `next_FS_pct_diff` | Same metric, but for the player's next match |
| `prev_minutes` | Duration of the immediately preceding match |
| `surface_experience` | Share of career matches played on current surface |
| `rank_diff` | Loser rank minus winner rank (size of the ranking gap) |

---

## Results at a Glance

| Hypothesis | Predictor | Direction | Significant? |
|---|---|---|---|
| H1 | Clutch index | Increases win probability | Yes |
| H2 | Recovery serving above average | Increases next win probability | Yes |
| H3 | Surface experience | Increases win probability | Yes |
| H3 (interaction) | Surface x rank gap | Dampens rank advantage | Yes |
| H4 | Previous match duration | No effect on clutch | No |

---

## Setup

```bash
pip install -r requirements.txt
jupyter notebook
```

No data download needed — `data_pipeline.ipynb` pulls directly from Jeff Sackmann's public GitHub at runtime.

---

## Dependencies

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

## Limitations Worth Noting

- Detailed match stats are only reliably available from around 2010 onward
- Everything is match-level — no point-by-point data, so I can't account for score state, weather, or crowd effects
- The clutch index gets noisy in short matches where a player faces very few break points (0s and 1s are common artifacts)
- The sequential analysis in H2 and H4 depends on matches being played within the same season
