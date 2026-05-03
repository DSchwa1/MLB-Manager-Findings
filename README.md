# Do Mid-Season Manager Firings Actually Work?

Fired teams improve more than matched controls—but not enough to rule out noise. After controlling for regression to the mean, the estimated effect of a mid-season managerial change on Pythagorean win percentage over the following 40 games is **+4.5 points** (p = 0.26, n = 45 events, 2001–2016).

---

## Why This Question Matters

The standard post-firing narrative—"the team responded to the change"—is almost always regression to the mean. Struggling teams fire managers when they're at their worst, and things tend to improve regardless. The real question for a front office is: **does firing the manager add anything beyond what the roster would have produced anyway?**

This project builds a matched control group to separate the manager signal from the mechanical recovery, then runs a regression to quantify the net effect.

---

## Key Findings

| Metric | Fired Teams | Matched Controls |
|---|---|---|
| Mean Pythagorean W% change (40-game post window) | +4.96 pts | +0.68 pts |
| Net estimated effect (β₁) | **+4.53 pts** | — |
| Standard error | ±3.99 pts | — |
| p-value | 0.26 | — |
| R² (full model) | 0.37 | — |

The direction is consistent: fired teams do improve more than controls in every robustness check (different post-window lengths, restricted pre-window samples). But the effect is imprecisely estimated—a sample of ~45 events over 25 years doesn't have the power to distinguish a real 4-point effect from noise at conventional significance thresholds.

**Secondary findings:**
- Outsider hires (external candidates) did not produce meaningfully different outcomes than insider promotions—the interaction term was small and non-significant
- Earlier firings (before game 80) showed no consistent advantage over later ones
- β₁ held its positive direction across all four robustness checks; it reached significance in none

**Practical implication:** The data is consistent with a modest real effect, but a front office shouldn't interpret the post-firing improvement as evidence that the decision worked. The improvement would likely have happened anyway, and the signal-to-noise ratio is too low to use this analysis prescriptively.

---

## Data Sources

| Data | Source | Coverage |
|---|---|---|
| Firing events | Lahman `Managers.csv` + Baseball Reference (verification) | 2001–2016 |
| Game logs | MLB Stats API (primary), pybaseball (fallback) | 2001–2016 |
| Preseason baseline | Prior-year Pythagorean W% via MLB Stats API | 2001–2016 |

> **Note on projections:** FanGraphs ZiPS projected standings are JavaScript-rendered and not programmatically scrapable. Phase 4 uses prior-year Pythagorean W% as the expectation baseline instead. The Lahman database (the primary event source) covers through 2016; 2017–2025 events are partially captured but not fully validated, so the final sample is scoped to 2001–2016.

---

## Methodology

### Event Identification
All mid-season managerial changes (April 1–September 30) where the team had at least 25 games remaining after the firing date. Events are identified from the Lahman `Managers.csv` `inseason` flag and cross-referenced against Baseball Reference manager pages to confirm firing vs. resignation and to capture the replacement manager.

### Performance Metric
Pythagorean W% (RS² / (RS² + RA²), exponent 1.83) rather than actual W%, because actual win percentage is heavily influenced by bullpen sequencing and close-game outcomes that don't reflect underlying team quality. The outcome variable `pyth_delta` is post-firing Pythagorean W% minus pre-firing Pythagorean W% over the 40 games immediately following the change.

### Control Group Construction
For each firing event, I identify a non-firing team from the same season at a comparable point (within ±15 games), with similar pre-period Pythagorean W% and projection residual. Each control team's improvement over an equivalent 40-game window serves as the regression-to-mean baseline. All 45 firing events were matched; no controls were reused across multiple events.

### Primary Regression

OLS on the pooled fired + control sample:

```
pyth_delta = β₀ + β₁(fired) + β₂(projection_residual_pre)
           + β₃(pyth_gap_at_firing) + β₄(game_number_at_firing) + ε
```

`fired` = 1 for managerial change events, 0 for matched controls. **β₁ is the headline estimate**—the average post-firing improvement in Pythagorean W% relative to matched controls, after accounting for pre-firing performance level, how far actual W% diverged from expected W% (luck indicator), and where in the season the firing occurred.

### Secondary Analyses
Four additional models examine whether the effect varies by hire type (outsider vs. insider), season timing, interim manager tenure length, and PA-weighted roster age at the time of firing.

### Robustness Checks
The primary regression is re-run with: (1) truncated post-window events excluded, (2) events restricted to ≥15 pre-firing games, (3) a 30-game post window, and (4) a 50-game post window. β₁ holds its positive direction across all four; it reaches significance in none.

---

## Limitations

- **Sample size:** ~2–4 mid-season firings per season produces ~45 events over 25 years. The confidence interval on β₁ spans from roughly −3 to +13 percentage points—wide enough that the data is consistent with both no effect and a meaningful one.
- **Projection baseline:** Prior-year Pythagorean W% is a reasonable proxy for preseason expectations but will be miscalibrated for teams with large offseason roster turnover.
- **Confounding roster moves:** Post-firing trades and call-ups are a plausible confound—front offices sometimes make complementary moves when they fire a manager. These are not controlled for.
- **`is_outsider` classification:** Automated from public records; all 45 values require manual verification before publication.
- **Event coverage (2017–2025):** The Lahman database's 2016 coverage cutoff means the 2017–2025 era relies on Baseball Reference scraping, which is more fragile. The final sample is effectively scoped to 2001–2016.

---

## How to Run

```bash
pip install -r requirements.txt

python run_analysis.py --all        # full pipeline, phases 0–10
python run_analysis.py --phase N    # re-run a single phase
```

All outputs are written to `outputs/` (gitignored; regenerated at runtime). Charts are written to `outputs/charts/`.

The pipeline logs every data exclusion and data-quality flag to `outputs/data_audit.txt`—check this file if any phase produces unexpected results.

---

## Repository Structure

```
├── run_analysis.py           # master runner; routes --phase N to correct module
├── utils.py                  # shared utilities: MLB Stats API, Pythagorean W%, audit logging
├── phase0_audit.py           # data availability audit
├── phase1_event_table.py     # event identification (firing events 2001–2016)
├── phase2_game_logs.py       # game-by-game records for pre/post windows
├── phase3_metrics.py         # Pythagorean W% and delta calculations
├── phase4_projections.py     # preseason expectation baseline
├── phase5_control_group.py   # matched control construction
├── phase6_regression.py      # primary OLS regression
├── phase7_secondary.py       # outsider/insider, timing, tenure, roster age models
├── phase8_robustness.py      # sensitivity checks (window length, sample restrictions)
├── phase9_visualizations.py  # matplotlib charts
├── phase10_summary.py        # summary findings output
└── requirements.txt
```

---

## Future Work

- **Extend coverage to 2017–2025:** Validate the BBRef scraping layer to recover the nine seasons currently outside the Lahman coverage window.
- **Control for roster transactions:** Flag post-firing trades and call-ups to test whether mid-season moves confound the improvement estimates.
- **Bayesian model:** With n=45, a hierarchical model with partial pooling would provide better-calibrated uncertainty than OLS and allow for more informative priors based on the broader coaching-change literature.
- **Mechanism analysis:** If the effect is real, does it operate through lineup decisions, bullpen usage, or something else? Play-by-play data could help decompose it.
- **Broader scope:** Apply the same matched-control framework to other managerial decisions—pitching coach changes, hitting coach changes, organizational philosophy shifts mid-season.
