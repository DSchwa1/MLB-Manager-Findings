# MLB Manager Firings — Claude Code Guide

## What this project is

Python analytics pipeline studying whether mid-season MLB managerial firings (2000–2025) produce genuine team improvement or just reflect regression to the mean. 11 sequential phases from data collection through regression, visualization, and summary.

## How to run

```bash
python run_analysis.py --all          # full pipeline, phases 0–10
python run_analysis.py --phase N      # re-run one phase (0–10)
```

## Phase map

| Phase | Script | Output |
|---|---|---|
| 0 | phase0_audit.py | outputs/data_audit.txt |
| 1 | phase1_event_table.py | outputs/event_table.csv |
| 2 | phase2_game_logs.py | outputs/game_log.csv |
| 3 | phase3_metrics.py | outputs/metrics_table.csv |
| 4 | phase4_projections.py | outputs/projections_table.csv |
| 5 | phase5_control_group.py | outputs/control_table.csv, control_pool.csv |
| 6 | phase6_regression.py | outputs/regression_primary.txt |
| 7 | phase7_secondary.py | outputs/regression_{outsider,timing,tenure,age}.txt |
| 8 | phase8_robustness.py | outputs/robustness_checks.txt |
| 9 | phase9_visualizations.py | outputs/charts/*.png |
| 10 | phase10_summary.py | outputs/summary_findings.txt |

## Key files

- `run_analysis.py` — master runner, routes `--phase N` to correct module
- `utils.py` — shared utilities: `bbref_get`, `get_mlb_game_log`, `get_prior_year_standings`, `pythagorean_wpct`, `log_audit`, team ID mappings
- `outputs/` — all generated files (gitignored, regenerated at runtime)

## Data sources

- **Firing events**: Baseball Reference (scraped) + Lahman DB via pybaseball
- **Game logs**: MLB Stats API (`statsapi.mlb.com`) primary, pybaseball fallback
- **Preseason baseline**: Prior-year Pythagorean W% from MLB Stats API standings (covers 1999–present; replaces FanGraphs ZiPS which is JS-rendered and not scrapable)

## Known issues / gotchas

- **Lahman DB download is broken** (`BadZipFile`) — pybaseball `teams_core()` etc. will fail. The pipeline works around this by using the MLB Stats API directly for all game log and standings data.
- **FanGraphs is JS-rendered** — all `.aspx` and API projection endpoints either 403 or return empty tables. Phase 4 was rewritten to avoid FanGraphs entirely.
- **BBRef 403s** — `bbref_get()` in utils.py uses a shared session, browser-like headers, 4s base delay + jitter, and a one-retry on 403. Still possible to get blocked on large scrapes.
- **`is_outsider` flag** — automated classification, needs manual verification for all events (see data_audit.txt).
- The `outputs/` folder is gitignored. Charts and CSVs are not committed — run the pipeline to regenerate.

## Primary outcome

`pyth_delta` = post-firing Pythagorean W% minus pre-firing Pythagorean W% (40-game post window).

Pythagorean W% used instead of actual W% to reduce sequencing/bullpen luck noise.

## Regression model (Phase 6)

```
pyth_delta = β0 + β1(fired) + β2(projection_residual_pre)
           + β3(pyth_gap_at_firing) + β4(game_number_at_firing) + ε
```

β1 is the headline: managerial change effect after controlling for regression pressure.
