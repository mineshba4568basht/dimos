# PR #1645 (nav1 / loop closure) — Paul Review Fixes

## Commits (local, not pushed)

### 1. `357a2b178` — Hoist scipy Rotation import to top-level
- Was imported inline 3 times in per-keyframe hot paths
- **Revert:** `git revert 357a2b178`

### 2. `a89b471f3` — Replace assert with explicit None check
- `assert` stripped in `python -O` — unsafe for production robot code
- **Revert:** `git revert a89b471f3`

### 3. `d0ffec6cf` — Fix timestamp falsy check
- `msg.ts=0.0` (epoch) is valid but falsy → incorrectly used `time.time()`
- Now uses `is not None`
- **Revert:** `git revert d0ffec6cf`

### 4. `e6837e425` — Vectorize column carve with np.isin
- Python loop over 50k+ points replaced with vectorized `np.isin`
- **Revert:** `git revert e6837e425`

## Not addressed (need Jeff's input)
- PGO file is 554 lines doing 4 things — split into algorithm.py + module.py?
- ICP + GTSAM optimization on subscriber thread — move to worker queue?
- Hardcoded camera offsets in PGO TF publishing — should be in config or camera module?
- Second GO2Connection in smartnav — intentional for publish_tf=False?
- Loop closure picks first KD-tree candidate, not best
