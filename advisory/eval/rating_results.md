# Blind rating of the heat advisories: results

Pre-registered in `context/decisions.md` (2026-10-30). Descriptive and post-test.
52 cases, 104 items, two raters. An item counts as unfaithful or harmful if either rater says so; scores 4-5 are the two raters' mean. 95% CIs: bootstrap over cases.

LLM generation outcomes (all cases and practice items): {'passed_first_try': 56}.

| Question | LLM (95% CI) | Rule-based (95% CI) | LLM minus rule-based (95% CI) | p |
|---|---|---|---|---|
| Faithful to the forecast (share of items) | 0.87 (0.77 to 0.94) | 0.92 (0.85 to 0.98) | -0.06 (-0.13 to +0.00) | 0.250 |
| Supported by the sources (share) | 0.00 (0.00 to 0.00) | 0.54 (0.40 to 0.67) | -0.54 (-0.67 to -0.40) | 0.000 |
| Harmful or misleading (share; lower is better) | 0.08 (0.02 to 0.15) | 0.00 (0.00 to 0.00) | +0.08 (+0.02 to +0.15) | 0.125 |
| Actions suit the level (mean, 1-5) | 3.72 (3.57 to 3.88) | 3.74 (3.58 to 3.90) | -0.01 (-0.11 to +0.08) | 1.000 |
| Clear for the public (mean, 1-5) | 4.22 (4.11 to 4.32) | 4.30 (4.21 to 4.40) | -0.08 (-0.21 to +0.05) | 0.207 |

Paired tests: McNemar (exact) for the yes/no questions, Wilcoxon signed-rank for the scores.

## Agreement between the raters

| Question | Kappa | Same answer |
|---|---|---|
| Faithful to the forecast (share of items) | 0.13 | 90% |
| Supported by the sources (share) | 1.00 | 100% |
| Harmful or misleading (share; lower is better) | 0.85 | 99% |
| Actions suit the level (mean, 1-5) | 0.72 | 83% |
| Clear for the public (mean, 1-5) | 0.19 | 42% |
