# MLB Mid-Season Manager Firings: Methodology

**Research question:** Do teams genuinely improve after a mid-season managerial change, or does the observed improvement reflect regression to the mean?

---

## Data Sources

| Data | Source | Coverage |
|---|---|---|
| Firing events | Baseball Reference (scraped) + Lahman DB via pybaseball | 2000–2025 |
| Game logs | MLB Stats API (primary), pybaseball fallback | 2000–2025 |
| Preseason projections | ZiPS via FanGraphs | 2006–2025 |
| Preseason projections | Marcel | 2000–2005 (unavailable — see Limitations) |

---

## Pipeline Overview

The analysis runs in 11 sequential phases via `run_analysis.py`.

```
python run_analysis.py --all          # run everything
python run_analysis.py --phase N      # re-run a single phase
```

| Phase | Script | Output |
|---|---|---|
| 0 | `phase0_audit.py` | `data_audit.txt` — data availability audit |
| 1 | `phase1_event_table.py` | `event_table.csv` — one row per firing event |
| 2 | `phase2_game_logs.py` | `game_log.csv` — pre/post game-by-game records |
| 3 | `phase3_metrics.py` | `metrics_table.csv` — per-event performance metrics |
| 4 | `phase4_projections.py` | `projections_table.csv` — preseason W% projections |
| 5 | `phase5_control_group.py` | `control_table.csv`, `control_pool.csv` |
| 6 | `phase6_regression.py` | `regression_primary.txt` |
| 7 | `phase7_secondary.py` | Four secondary regression files |
| 8 | `phase8_robustness.py` | `robustness_checks.txt` |
| 9 | `phase9_visualizations.py` | PNG charts in `outputs/charts/` |
| 10 | `phase10_summary.py` | `summary_findings.txt` |

---

## Event Identification (Phase 1)

A firing event is defined as a mid-season managerial change between April 1 and September 30. Events are sourced from Baseball Reference manager pages and cross-referenced against the Lahman database.

For each event the pipeline records:
- Team, season, date of firing
- Fired manager and replacement
- Game number at the time of firing
- Whether the replacement was an outsider hire (`is_outsider`)

`is_outsider` is automated based on prior MLB managerial experience but is flagged for manual verification in `data_audit.txt`.

---

## Performance Windows (Phases 2–3)

For each firing event, two windows are extracted from the full-season game log:

- **Pre window:** All games played up to and including the day of firing
- **Post window:** The 40 games immediately following the firing

Two performance metrics are computed for each window:

**Pythagorean W%** — expected win percentage based on runs scored and allowed:

$$\hat{W}\% = \frac{RS^2}{RS^2 + RA^2}$$

**Actual W%** — observed wins / (wins + losses), excluding ties.

Key derived metrics:
- `pyth_delta` — post Pythagorean W% minus pre Pythagorean W% (primary outcome)
- `pyth_gap_at_firing` — pre actual W% minus pre Pythagorean W% (luck/sequencing indicator)
- `run_diff_per_game` — (RS − RA) / games played

Pythagorean W% is used as the primary outcome rather than actual W% because it is less sensitive to bullpen sequencing and luck, giving a cleaner signal of underlying team quality.

---

## Preseason Projections (Phase 4)

Projected W% is pulled from ZiPS (FanGraphs) for 2006–2025 and used to estimate how much of a team's pre-firing performance was expected vs. a surprise.

`projection_residual_pre` = actual pre W% − projected W%

A negative residual means the team was underperforming expectations at the time of the firing.

Marcel projections (2000–2005) were never published in a consistently archived machine-readable format and are unavailable. `projected_wpct` is null for those seasons.

---

## Control Group (Phase 5)

To isolate the managerial change effect from regression to the mean, each firing event is matched to a control observation: a non-firing team at a similar point in a similar season.

**Control pool construction:** For every season in the study period, all 30 active teams are enumerated. Any team that fired its manager that season is excluded. For each remaining team, a 40-game window is extracted at the same game number as each firing event in that season.

**Matching variables:**
- Pre-window Pythagorean W%
- Pythagorean gap at pseudo-firing date
- Projection residual (where available)

Each firing event is matched to the closest control observation by Euclidean distance across these variables.

---

## Primary Regression (Phase 6)

OLS regression on the combined fired + matched-control sample:

$$\text{pyth\_delta} = \beta_0 + \beta_1(\text{fired}) + \beta_2(\text{projection\_residual\_pre}) + \beta_3(\text{pyth\_gap\_at\_firing}) + \beta_4(\text{game\_number\_at\_firing}) + \varepsilon$$

**Interpretation of β₁:** The coefficient on `fired` is the primary estimate of the managerial change effect — the average difference in post-firing Pythagorean improvement between teams that fired their manager and matched controls, after controlling for pre-firing performance, projection residual, and season timing.

---

## Secondary Analyses (Phase 7)

Four additional models examine moderating factors:

1. **Outsider vs. insider** — does bringing in an outside hire produce a different effect than promoting from within?
2. **Timing** — does firing earlier or later in the season predict a larger improvement?
3. **Interim tenure** — does the length of the interim manager's run moderate the effect?
4. **Roster age** — does PA-weighted average roster age at the time of firing moderate the effect?

---

## Robustness Checks (Phase 8)

The primary regression is re-run four times with modified samples or windows to test stability:

1. Exclude events with a truncated post window
2. Restrict to events with at least 15 pre-firing games
3. Use a 30-game post window instead of 40
4. Use a 50-game post window instead of 40

Results note whether β₁ holds its direction and significance across all four checks.

---

## Known Limitations

- **Marcel projections (2000–2005):** Not available in scrapable form. `projected_wpct` is null for these seasons; projection-residual terms are excluded from models for those events.
- **ZiPS historical coverage (2006–2014):** Some earlier FanGraphs projected-standings pages may be missing or require manual retrieval.
- **`is_outsider` classification:** Automated based on prior MLB managerial records; all values are flagged for manual review.
- **Sample size:** Mid-season firings are rare (~2–4 per season), limiting statistical power. Results should be interpreted with appropriate uncertainty.
- **Post-window truncation:** Firings late in the season may not have 40 games remaining; truncated events are flagged and sensitivity-tested in Phase 8.
