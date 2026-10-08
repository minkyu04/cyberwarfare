# Final Paper Result Tables

## Table 1. Overall A–F security performance
| Condition | Unauthorized evidence emission | Unauthorized retrieval exposure | Forbidden disclosure | Overblocking | Authorized fact recall | Authorized utility retention | Internal weighted utility retention |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 214/352 (60.80%; 95% CI 55.61–65.75) | 240/352 (68.18%; 95% CI 63.14–72.83) | 190/352 (53.98%; 95% CI 48.76–59.11) | 24/101 (23.76%; 95% CI 16.52–32.93) | 76.73% (95% CI 68.51–84.96) | 77.03% (95% CI 68.82–85.24) |  |
| B | 217/352 (61.65%; 95% CI 56.47–66.58) | 240/352 (68.18%; 95% CI 63.14–72.83) | 172/352 (48.86%; 95% CI 43.68–54.07) | 24/101 (23.76%; 95% CI 16.52–32.93) | 77.23% (95% CI 69.12–85.33) | 77.82% (95% CI 69.75–85.89) |  |
| C | 11/352 (3.12%; 95% CI 1.75–5.51) | 0/352 (0.00%; 95% CI 0.00–1.08) | 34/352 (9.66%; 95% CI 6.99–13.19) | 22/101 (21.78%; 95% CI 14.85–30.78) | 79.21% (95% CI 71.37–87.04) | 79.80% (95% CI 72.01–87.59) |  |
| D | 9/352 (2.56%; 95% CI 1.35–4.79) | 0/352 (0.00%; 95% CI 0.00–1.08) | 28/352 (7.95%; 95% CI 5.56–11.26) | 20/101 (19.80%; 95% CI 13.20–28.62) | 81.19% (95% CI 73.65–88.72) | 81.78% (95% CI 74.30–89.27) |  |
| E | 0/352 (0.00%; 95% CI 0.00–1.08) | 0/352 (0.00%; 95% CI 0.00–1.08) | 30/352 (8.52%; 95% CI 6.04–11.91) | 29/101 (28.71%; 95% CI 20.80–38.19) | 72.28% (95% CI 63.61–80.94) | 72.87% (95% CI 64.23–81.52) |  |
| F | 0/352 (0.00%; 95% CI 0.00–1.08) | 0/352 (0.00%; 95% CI 0.00–1.08) | 0/352 (0.00%; 95% CI 0.00–1.08) | 0/101 (0.00%; 95% CI 0.00–3.66) | 100.00% (95% CI 100.00–100.00) | 100.00% (95% CI 100.00–100.00) | 89.34% (95% CI 86.23–92.45) |

## Table 2. AT7 and AT8 security performance
| Condition | AT7 cross-document | AT8 cumulative | AT8 post-revocation |
| --- | --- | --- | --- |
| A | 23/32 (71.88%; 95% CI 54.63–84.44) | 4/8 (50.00%; 95% CI 21.52–78.48) | 5/8 (62.50%; 95% CI 30.57–86.32) |
| B | 19/32 (59.38%; 95% CI 42.26–74.48) | 4/8 (50.00%; 95% CI 21.52–78.48) | 2/8 (25.00%; 95% CI 7.15–59.07) |
| C | 20/32 (62.50%; 95% CI 45.25–77.07) | 4/8 (50.00%; 95% CI 21.52–78.48) | 4/8 (50.00%; 95% CI 21.52–78.48) |
| D | 18/32 (56.25%; 95% CI 39.33–71.83) | 4/8 (50.00%; 95% CI 21.52–78.48) | 5/8 (62.50%; 95% CI 30.57–86.32) |
| E | 17/32 (53.12%; 95% CI 36.45–69.13) | 4/8 (50.00%; 95% CI 21.52–78.48) | 3/8 (37.50%; 95% CI 13.68–69.43) |
| F | 0/32 (0.00%; 95% CI 0.00–10.72) | 0/8 (0.00%; 95% CI 0.00–32.44) | 0/8 (0.00%; 95% CI 0.00–32.44) |

## Table 3. Ablation analysis
| Configuration | n | Forbidden disclosure rate | Cumulative leak rate | Overblocking rate | Authorized fact recall | Authorized utility retention | Internal weighted utility retention |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full F | 88 | 0.00% | 0.00% | 0.00% | 100.00% | 100.00% | 81.56% |
| F without history | 88 | 4.55% | 4.55% | 0.00% | 100.00% | 100.00% | 91.31% |
| F without utility weights | 88 | 2.27% | 0.00% | 16.67% | 83.33% | 83.33% | 78.20% |
| F all-block | 88 | 0.00% | 0.00% | 100.00% | 0.00% | 0.00% | 0.00% |

## Table 4. Cross-model robustness
| Model | Condition | AT7 cross-document | AT8 cumulative | AT8 post-revocation |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | C | 20/32 (62.50%; 95% CI 45.25–77.07) | 4/8 (50.00%; 95% CI 21.52–78.48) | 4/8 (50.00%; 95% CI 21.52–78.48) |
| Qwen2.5-0.5B | E | 17/32 (53.12%; 95% CI 36.45–69.13) | 4/8 (50.00%; 95% CI 21.52–78.48) | 3/8 (37.50%; 95% CI 13.68–69.43) |
| Qwen2.5-0.5B | F | 0/32 (0.00%; 95% CI 0.00–10.72) | 0/8 (0.00%; 95% CI 0.00–32.44) | 0/8 (0.00%; 95% CI 0.00–32.44) |
| Qwen2.5-3B | C | 32/32 (100.00%; 95% CI 89.28–100.00) | 8/8 (100.00%; 95% CI 67.56–100.00) | 8/8 (100.00%; 95% CI 67.56–100.00) |
| Qwen2.5-3B | E | 32/32 (100.00%; 95% CI 89.28–100.00) | 8/8 (100.00%; 95% CI 67.56–100.00) | 8/8 (100.00%; 95% CI 67.56–100.00) |
| Qwen2.5-3B | F | 0/32 (0.00%; 95% CI 0.00–10.72) | 0/8 (0.00%; 95% CI 0.00–32.44) | 0/8 (0.00%; 95% CI 0.00–32.44) |
| Phi-3.5-mini | C | 30/32 (93.75%; 95% CI 79.85–98.27) | 7/8 (87.50%; 95% CI 52.91–97.76) | 8/8 (100.00%; 95% CI 67.56–100.00) |
| Phi-3.5-mini | E | 30/32 (93.75%; 95% CI 79.85–98.27) | 7/8 (87.50%; 95% CI 52.91–97.76) | 8/8 (100.00%; 95% CI 67.56–100.00) |
| Phi-3.5-mini | F | 0/32 (0.00%; 95% CI 0.00–10.72) | 0/8 (0.00%; 95% CI 0.00–32.44) | 0/8 (0.00%; 95% CI 0.00–32.44) |

## Table 5. Utility-weight sensitivity
| Perturbation | Trials | Safety preserved | Exact selection agreement | Baseline-optimal equivalent | Changed but equivalent | Strict rank reversal | Mean rank-reversal pair rate | Mean baseline regret / total utility | Mean original utility retention |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ±10% | 48000 | 100.00% | 62.64% | 100.00% | 37.36% | 0.00% | 0.00% | 0.00% | 55.68% |
| ±20% | 48000 | 100.00% | 62.69% | 100.00% | 37.31% | 0.00% | 0.00% | 0.00% | 55.68% |
| ±30% | 48000 | 100.00% | 62.45% | 100.00% | 37.55% | 0.00% | 0.00% | 0.00% | 55.68% |
| ±50% | 48000 | 100.00% | 62.72% | 99.96% | 37.24% | 1.57% | 0.53% | 0.02% | 55.66% |

## Table 6. Paired exact tests
| Model | Metric | Comparison | Paired n | A0/B1 | A1/B0 | Exact two-sided p |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B | Overall forbidden disclosure | C vs F | 352 | 0 | 34 | 1.16415e-10 |
| Qwen2.5-0.5B | Overall forbidden disclosure | E vs F | 352 | 0 | 30 | 1.86265e-09 |
| Qwen2.5-0.5B | AT7 cross-document | C vs F | 32 | 0 | 20 | 1.90735e-06 |
| Qwen2.5-0.5B | AT7 cross-document | E vs F | 32 | 0 | 17 | 1.52588e-05 |
| Qwen2.5-0.5B | AT8 cumulative | C vs F | 8 | 0 | 4 | 0.125 |
| Qwen2.5-0.5B | AT8 cumulative | E vs F | 8 | 0 | 4 | 0.125 |
| Qwen2.5-0.5B | AT8 post-revocation | C vs F | 8 | 0 | 4 | 0.125 |
| Qwen2.5-0.5B | AT8 post-revocation | E vs F | 8 | 0 | 3 | 0.25 |
| Qwen2.5-3B | Overall forbidden disclosure | C vs F | 64 | 0 | 48 | 7.10543e-15 |
| Qwen2.5-3B | Overall forbidden disclosure | E vs F | 64 | 0 | 48 | 7.10543e-15 |
| Qwen2.5-3B | AT7 cross-document | C vs F | 32 | 0 | 32 | 4.65661e-10 |
| Qwen2.5-3B | AT7 cross-document | E vs F | 32 | 0 | 32 | 4.65661e-10 |
| Qwen2.5-3B | AT8 cumulative | C vs F | 8 | 0 | 8 | 0.0078125 |
| Qwen2.5-3B | AT8 cumulative | E vs F | 8 | 0 | 8 | 0.0078125 |
| Qwen2.5-3B | AT8 post-revocation | C vs F | 8 | 0 | 8 | 0.0078125 |
| Qwen2.5-3B | AT8 post-revocation | E vs F | 8 | 0 | 8 | 0.0078125 |
| Phi-3.5-mini | Overall forbidden disclosure | C vs F | 64 | 0 | 46 | 2.84217e-14 |
| Phi-3.5-mini | Overall forbidden disclosure | E vs F | 64 | 0 | 46 | 2.84217e-14 |
| Phi-3.5-mini | AT7 cross-document | C vs F | 32 | 0 | 30 | 1.86265e-09 |
| Phi-3.5-mini | AT7 cross-document | E vs F | 32 | 0 | 30 | 1.86265e-09 |
| Phi-3.5-mini | AT8 cumulative | C vs F | 8 | 0 | 7 | 0.015625 |
| Phi-3.5-mini | AT8 cumulative | E vs F | 8 | 0 | 7 | 0.015625 |
| Phi-3.5-mini | AT8 post-revocation | C vs F | 8 | 0 | 8 | 0.0078125 |
| Phi-3.5-mini | AT8 post-revocation | E vs F | 8 | 0 | 8 | 0.0078125 |

## Interpretation cautions

- Qwen2.5-0.5B full-test results are the primary experiment.
- Qwen2.5-3B and Phi-3.5-mini results are focused AT7/AT8 robustness checks rather than full A–F replications.
- AT8 cumulative and post-revocation scopes contain eight evaluated second-turn cases per condition in the test split; confidence intervals are therefore wide.
- Zero observed events should not be described as proof of zero deployment risk.
- LLM generation latency is not used for cross-condition performance claims because the generation path differs when the proposed controller blocks or reduces response content.
- Utility sensitivity intervals summarize Monte Carlo perturbation observations over the fixed synthetic rule set; they are not deployment-population confidence intervals.
