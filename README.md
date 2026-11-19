# gta6-rarity-forecast-f

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/contact-nathanealphi/gta6-rarity-forecast-f/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/contact-nathanealphi/gta6-rarity-forecast-f/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Cohort E/N forecast of GTA VI PS5 trophy earn rates (Exophase) at 26 Nov 2026 00:00Z: one Python 3.13 `main.py`, stdlib only, no network. It models earners and the tracked-player population separately — rate = E(t)/N(t) — with per-trophy play-hour thresholds, a refresh-aware page state (the scored answer is the stats page state, which only changes at refreshes), and a dilution-plus-progression cohort simulation: the old cohort progresses toward its own asymptote, each new batch arrives at a diluted rate that rises as the player base matures, and the population grows on a fitted power law (capped at 5x). Spreads are wide and symmetric in logit space.

## Run it

```sh
python3 main.py
```

Reads every `*.in` in `$MAKEFUN_INPUT_DIR` (else `/input/`), writes `$MAKEFUN_OUTPUT_DIR/<id>.out`. Deterministic: two runs are byte-identical. About 0.05–0.2 s per case.

---

Entered on [make.fun](https://make.fun/?ref=9ncgmbd6&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards
