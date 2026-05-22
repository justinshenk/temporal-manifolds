# Roadmap

Mirror of the [Temporal Manifolds Google Sheet](https://docs.google.com/spreadsheets/d/1hyTdfxwrIedeaYX2Ri5xhSj5VCqW0T4EW90J2H6s6p0/edit?gid=0#gid=0)
(snapshot 2026-05-22). Update this file when the sheet changes.

## Workshop track (BlackboxNLP, submission 2026-07-17)

| #  | Category              | Task                                                              | Member            | Status        | Start       | End         | Output                                                                 |
|----|-----------------------|-------------------------------------------------------------------|-------------------|---------------|-------------|-------------|------------------------------------------------------------------------|
| 01 | Parametric Dataset    | Generate parametric dataset across multiple scenarios             | Shantanu          | In progress   | 2026-05-20  | 2026-05-23  | Algorithmic dataset generation code; config-driven scenarios           |
| 02 | Activation Caching    | Cache node activations at 5 key positions on each dataset         | Shantanu, Justin  | Not started   | 2026-05-23  | 2026-05-25  | Activation tensors pushed to HF                                        |
| 03 | Geometric Modeling    | Replicate plots from arxiv:2605.05115 on our datasets             | —                 | Not started   | 2026-05-25  | 2026-06-02  | A pair of plots for each scenario                                      |
| 04 | Constraint Phrasing   | Quantify alignment between equivalent temporal phrasings          | —                 | Not started   | 2026-06-02  | 2026-06-07  | Metric + hypothesis explaining observations                            |
| —  | Workshop Paper        | Begin paper writing                                                | —                 | Not started   | 2026-06-07  | 2026-06-18  | Paper ready for expert feedback                                        |
| —  | Feedback from Experts | Reach out to experts and request feedback                          | —                 | Not started   | 2026-06-11  | 2026-06-30  | Feedback (via professional contacts)                                   |
| —  | Workshop Paper        | Finalize paper based on feedback                                   | —                 | Not started   | 2026-07-01  | 2026-07-10  | Final paper for submission                                             |

## Conference track (post-workshop)

| #  | Category                | Task                                                               | Output                                                                                |
|----|-------------------------|--------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| 05 | Linear Baseline         | Linear regression on residual activations → planning horizon        | Baseline model for comparison                                                         |
| 06 | Scenario Consistency    | PCA per scenario; observe manifold shapes                           | A plot per scenario, points colored by log-time-horizon                               |
| 06 | Scenario Consistency    | Align scenario-specific manifolds in a shared space                 | Joint plot + hypothesis correlating observations with prompt features                 |
| 07 | Predictive Model        | Predictive model over concatenated activations → planning horizon   | Trained predictor; objective may include phrasing-equivalence distance                |
| 08 | Predictive Power        | Evaluate on held-out parametric prompts                             | Improvement vs. linear baseline                                                       |
| 08 | Predictive Power        | Compare to existing evals (arxiv:2509.15541)                        | Comparison table                                                                      |
| 09 | Cross-model generalization | Repeat on other models                                           | Cross-model results (some conferences require this)                                   |

## Notes

- Hold-out split should be by **phrasing group**, not by row — we want to
  measure whether unseen phrasings collapse onto the same point.
- Activation caching positions: 5 specific token positions identified in prior
  runs (the Qwen3-32B steering work). Confirm and document them in
  `experiments/02_activation_caching/README.md`.
- For experiment 03: reference code lives at
  `goodfire-ai/causalab` (branch `manifold_steering`).
