# Presentation Explanations for DDoS Detection Results

This guide explains the generated result plots and tables in presentation-ready language. It is written to accompany the artifacts in `outputs/plots/` and `outputs/tables/`.

It intentionally does not list every individual file in `outputs/plots/confusion_matrices/`; those are treated as one family of supporting plots.

## One-Slide Model Architecture Summary

The project evaluates DDoS detection at three derived levels:

| Level | Meaning | How the prediction is produced |
| --- | --- | --- |
| IP instance | One `(src_ip, window_id)` pair | The model directly scores each active source IP inside each time window. |
| Window | One temporal sub-window of a 15-minute capture | A window is flagged if the maximum IP-instance score in that window exceeds the calibrated threshold. |
| Dataset | One full 15-minute CSV | A dataset is flagged if any window is flagged. |

The central model is the **Quantum ZZ** pipeline:

1. Raw flows are partitioned into fixed windows using `window_id = row_in_window // window_width`.
2. Each active source IP in each window becomes one training instance.
3. The instance is represented by 33 engineered network-traffic features: traffic volume, packet/byte rates, duration, packet size, timing, protocol mix, destination concentration, and cross-IP co-targeting features.
4. Features are log-scaled where appropriate, standardized, projected to `k = 8` PCA components, and mapped to angles with `theta = 2 * arctan(z)`.
5. The angles feed a fixed ZZ feature map: 8 qubits, 2 circuit repetitions, linear nearest-neighbor ZZ interactions.
6. The quantum state is converted back into classical features by measuring `Z_j` and adjacent `Z_j Z_{j+1}` expectation values. For `k = 8`, this gives `2k - 1 = 15` quantum features.
7. A single linear layer maps those 15 values to a malicious probability. The layer has only 16 trainable parameters: 15 weights plus 1 bias.
8. The operating threshold is calibrated on validation F1: `0.52` for `family_a` and `0.72` for `family_b`.

Presentation wording:

> The architecture is intentionally small. The domain knowledge is in the causal window-level feature engineering, while the quantum-inspired component acts as a fixed nonlinear feature map. The only learned component is a calibrated linear scorer.

Configurable quantum alternatives:

- `zz_linear`: the original fixed ZZ feature map plus a small linear scorer.
- `vqc`: a trainable differentiable VQC. The same PCA angles are encoded into a statevector circuit with trainable `RY` rotations and adjacent `RZZ` phases before Z/ZZ readout.
- `quantum_kernel`: a pure quantum kernel mode. The ZZ map returns exact statevectors and the classifier is trained on `K(x, y) = |<phi(x)|phi(y)>|^2`.

Presentation wording:

> The pipeline can now separate the question “what quantum representation do we use?” from the rest of the DDoS detector. Windowing, feature engineering, preprocessing, calibration, and evaluation stay fixed while the quantum architecture is switched in the config.

## Models and Baselines

| Model | Architecture | Role in the presentation |
| --- | --- | --- |
| Quantum ZZ | 33 engineered features -> log scaling -> StandardScaler -> PCA to 8 dimensions -> ZZ feature map -> 15 expectation values -> linear scorer | Main proposed model. It is interpretable, small, and online because it operates per IP per window. |
| Majority | Always predicts benign | Floor baseline. It shows what happens when the detector never raises alerts. |
| Classical RBF | Same windowed instance features -> StandardScaler -> random Fourier RBF features -> balanced LogisticRegression | Direct classical kernel analogue to the quantum feature-map idea. |
| PCA Linear | StandardScaler -> PCA to 8 dimensions -> linear PyTorch scorer | Ablation. It tests whether PCA alone is enough without the ZZ nonlinear feature map. |
| Bag RBF | Whole 15-minute dataset aggregated per source IP -> RBF LogisticRegression | Reference/ceiling baseline. It is useful for dataset-level sanity checks, but it is not the same online detector because it removes temporal windowing. |

Important caveat:

> Bag RBF should not be presented as an operational window detector. It uses whole-capture aggregation, so it is a ceiling/reference model rather than the same streaming task.

## Executive Summary Plot and Table

Artifacts:

- `outputs/plots/presentation_quantum_executive_summary.png`
- `outputs/tables/quantum_executive_summary.csv`
- `outputs/tables/quantum_executive_summary.md`

What it shows:

This is the most compact view of the proposed Quantum ZZ pipeline. It reports the calibrated threshold, validation F1, test instance F1/AUC, test window F1/AUC, dataset F1, mean detection latency, and missed bursts for each family.

Results to say out loud:

| Family | Threshold | Validation F1 | Test instance F1 | Test instance AUC | Test window F1 | Test window AUC | Missed bursts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| family_a | 0.52 | 0.583 | 0.674 | 0.810 | 0.462 | 1.000 | 0 |
| family_b | 0.72 | 0.347 | 0.281 | 0.859 | 0.471 | 0.682 | 0 |

Interpretation:

Family A is the cleaner result for the quantum model: instance-level F1 reaches `0.674`, and window-level AUC is perfect at `1.000`. Family B is harder at the IP attribution level: instance F1 drops to `0.281`, even though instance AUC remains fairly strong at `0.859`. This means the raw ranking signal still separates attack from benign examples, but the chosen operating threshold produces many false positives and misses some malicious IPs.

Presentation takeaway:

> The quantum pipeline detects attack windows with zero burst misses in both families, but its calibrated operating point is recall-oriented. It is better at raising timely alerts than at producing a clean malicious-IP shortlist, especially in Family B.

## Test F1 Comparison Plot

Artifact:

- `outputs/plots/presentation_test_f1_comparison.png`

What it shows:

This grouped bar chart compares test F1 across methods and prediction levels: IP instance, window, and dataset.

Main results:

- In Family A, Quantum ZZ reaches `0.674` instance F1 and `0.462` window F1.
- In Family B, Quantum ZZ reaches `0.281` instance F1 and `0.471` window F1.
- PCA Linear is strongest among the windowed online models in both families: Family A window F1 `0.857`, Family B window F1 `0.941`.
- Bag RBF reaches very high scores, but it is not directly comparable as an online model because it uses whole-dataset aggregation.

Interpretation:

The PCA Linear ablation outperforming Quantum ZZ is an important scientific result. It suggests that, for the current feature set and dataset, much of the discriminative structure is already linearly accessible after engineered features and PCA. The ZZ feature map does not yet add enough useful nonlinearity to beat the simpler PCA-linear scorer.

Presentation takeaway:

> The quantum-inspired model is functional and highly compact, but the ablation shows that this dataset is already very separable with classical PCA features. That makes PCA Linear the strongest online baseline here.

## Test Metrics Tables

Artifacts:

- `outputs/plots/presentation_test_metrics_table.png`
- `outputs/plots/presentation_family_a_test_metrics_table.png`
- `outputs/plots/presentation_family_b_test_metrics_table.png`
- `outputs/tables/test_metrics_presentation.csv`
- `outputs/tables/test_metrics_presentation.md`

What they show:

These tables provide the exact test precision, recall, F1, and AUC behind the bar charts. The all-family table is useful as an appendix slide; the family-specific plots are easier to read in a main presentation.

Family A key points:

- Quantum ZZ IP-instance precision `0.603`, recall `0.763`, F1 `0.674`, AUC `0.810`.
- Quantum ZZ window precision `0.300`, recall `1.000`, F1 `0.462`, AUC `1.000`.
- PCA Linear is substantially stronger: IP-instance F1 `0.958`, window F1 `0.857`.
- Classical RBF is also stronger than Quantum ZZ at the instance level: F1 `0.766`.

Family B key points:

- Quantum ZZ IP-instance precision `0.205`, recall `0.450`, F1 `0.281`, AUC `0.859`.
- Quantum ZZ window precision `0.308`, recall `1.000`, F1 `0.471`, AUC `0.682`.
- PCA Linear and Bag RBF are much stronger in Family B, with PCA Linear window F1 `0.941` and Bag RBF instance F1 `0.974`.

Presentation wording:

> The quantum model prioritizes recall at the window level. In operational terms, that means it is unlikely to miss attack windows, but it raises too many benign windows as attacks. The classical ablations show that thresholding and representation choice are the main opportunities for improvement.

## Family Metric Heatmaps

Artifacts:

- `outputs/plots/presentation_family_a_test_metric_heatmap.png`
- `outputs/plots/presentation_family_b_test_metric_heatmap.png`

What they show:

Each heatmap gives a dense visual comparison of precision, recall, F1, and AUC across models and output levels for one family.

How to read them:

- Darker colors indicate stronger performance.
- Rows combine a method and an evaluation level.
- The heatmap is best for spotting patterns rather than quoting exact numbers.

Family A interpretation:

Family A shows a clear separation between the simple online ablation and the quantum model. PCA Linear is strong across IP and window levels. Quantum ZZ has full window recall but low window precision, which produces a moderate window F1.

Family B interpretation:

Family B is more challenging for Quantum ZZ. It still catches all attack windows at the chosen threshold, but precision is weak and the AUC at the window level is only `0.682`. This indicates that the score ordering is less stable for native DDoS traffic than for the recast DoS-as-DDoS family.

Presentation takeaway:

> The heatmaps make the main pattern visible: the engineered features are very useful, but the fixed ZZ map is not the best classifier for these two current families.

## Quantum Confusion Matrix Grids

Artifacts:

- `outputs/plots/presentation_family_a_quantum_confusion_matrices.png`
- `outputs/plots/presentation_family_b_quantum_confusion_matrices.png`

What they show:

Each grid shows the Quantum ZZ confusion matrices for the test split at the IP-instance, window, and dataset levels. Rows are true labels; columns are predictions.

Family A results:

- IP instance: `522` true positives, `162` false negatives, `344` false positives, `1181` true negatives.
- Window: `9` true-positive attack windows, `0` missed attack windows, `21` false-positive benign windows.
- Dataset: both attack datasets are detected, but both normal datasets are also flagged.

Family B results:

- IP instance: `36` true positives, `44` false negatives, `140` false positives, `1360` true negatives.
- Window: `8` true-positive attack windows, `0` missed attack windows, `18` false-positive benign windows, `4` true-negative benign windows.
- Dataset: both attack datasets are detected, but both normal datasets are also flagged.

Interpretation:

The window-level confusion matrices are the clearest expression of the detector tradeoff. The quantum model avoids false negatives at the window level, so it works as an early warning mechanism. The cost is many false positives, especially in Family A where all benign windows in the attack-only window evaluation are crossed by the threshold.

Presentation wording:

> The confusion matrices show that the model is tuned conservatively: it catches every attack window but pays for that with false alerts. This is often acceptable for a first-stage detector, but a production system would need a second-stage triage model or a stricter threshold policy.

## Individual Confusion Matrix Plot Family

Artifact family:

- `outputs/plots/confusion_matrices/*.png`

What it contains:

This directory contains one confusion-matrix PNG for each combination of family, method, split, and evaluation level.

How to use it:

Use these as appendix or backup slides when someone asks for a specific method-level breakdown. Do not put all of them in the main deck. The main deck should use only the two Quantum ZZ grids plus selected supporting matrices when needed.

Recommended selections:

- Use Quantum ZZ test matrices to explain the main model.
- Use PCA Linear test matrices if you want to explain why the ablation is stronger.
- Use Majority matrices only to show the floor baseline.

Presentation takeaway:

> The individual matrices are diagnostic artifacts. They are useful for auditability, but the presentation should focus on the summary grids and rankings.

## Detection Latency Plot and Table

Artifacts:

- `outputs/plots/presentation_detection_latency.png`
- `outputs/tables/detection_latency_summary.csv`
- `outputs/tables/detection_latency_summary.md`

What they show:

Detection latency is measured in windows after the first ramp-up window of a burst. A latency of `0` means the detector flags the burst in the ramp-up window itself.

Key results:

- Quantum ZZ detects all measured bursts with mean latency `0.0` windows in both families.
- Classical RBF and PCA Linear also detect bursts at latency `0.0` in the reported online window setting.
- Majority misses all bursts.
- Bag RBF has no meaningful online latency because it uses whole-capture aggregation rather than streaming windows.

Presentation wording:

> The most operationally important result is that the online detectors do not wait until after the attack has fully developed. They trigger during the ramp-up window. The remaining problem is alert precision, not initial detection speed.

## Threshold Plot and Table

Artifacts:

- `outputs/plots/presentation_thresholds.png`
- `outputs/tables/threshold_summary.csv`
- `outputs/tables/threshold_summary.md`
- `outputs/threshold_family_a.json`
- `outputs/threshold_family_b.json`

What they show:

These artifacts show the validation-calibrated probability threshold used for the quantum model.

Results:

- Family A threshold: `0.52`, validation F1 `0.583`.
- Family B threshold: `0.72`, validation F1 `0.347`.

Interpretation:

Family B needs a higher threshold but still has lower validation F1. That means the Family B score distribution is harder to calibrate: benign and malicious instances overlap more strongly, even if the AUC suggests some ranking signal remains.

Presentation takeaway:

> Threshold calibration makes the model operational, but the lower Family B validation F1 shows that one global threshold per family is not enough to fully solve IP attribution.

## Temporal Detection Plot Family

Artifact family:

- `outputs/plots/temporal_family_a_attack_test_01.png`
- `outputs/plots/temporal_family_a_attack_test_02.png`
- `outputs/plots/temporal_family_b_attack_test_01.png`
- `outputs/plots/temporal_family_b_attack_test_02.png`

What they show:

Each plot is an attack test dataset over time. The x-axis is the window index. The y-axis is the maximum malicious probability among active source IPs in that window. Shaded vertical regions are true burst windows. The dashed horizontal line is the calibrated threshold.

How to read them:

- A good detector rises above the threshold inside or near shaded burst regions.
- A window above the threshold is predicted as an attack window.
- Shading marks ground-truth burst windows, not model predictions.
- A false positive occurs when an unshaded window crosses the threshold.
- A missed burst occurs when a shaded region stays below the threshold.

Current result interpretation:

The aggregate latency metrics show no missed bursts for Quantum ZZ, so the temporal plots should show threshold crossings at burst onset. The confusion matrices also show false-positive windows, so these plots are the best place to visually inspect whether the false positives are isolated spikes or sustained alert periods.

Important clarification:

> If the dashed threshold is below the max-score line but the interval is not shaded, the model did predict an attack for that interval. The interval is unshaded because it is not a ground-truth burst window. In metric terms, that is a false-positive window.

Presentation wording:

> The temporal plots convert the metrics into an operational timeline. They show that the detector can raise alerts during attack activity, while also revealing where alert smoothing or a second-stage filter would reduce false positives.

## Temporal Comparison Plot Family

Artifact family:

- `outputs/plots/temporal_compare_{quantum_architecture}_family_a_attack_test_01.png`
- `outputs/plots/temporal_compare_{quantum_architecture}_family_a_attack_test_02.png`
- `outputs/plots/temporal_compare_{quantum_architecture}_family_b_attack_test_01.png`
- `outputs/plots/temporal_compare_{quantum_architecture}_family_b_attack_test_02.png`
- `outputs/tables/temporal_comparison_summary.csv`
- `outputs/tables/temporal_comparison_summary.md`

How they were produced:

These plots are generated by `python temporal_compare.py`. The script uses the quantum architecture selected by `config.json` (`zz_linear`, `vqc`, or `quantum_kernel`) and compares it with the best online/windowed model for each family using test window F1 from `outputs/results_{family}.json`. It intentionally excludes Bag RBF because Bag RBF aggregates the whole 15-minute capture and is not a streaming window detector.

Current selected model:

- Family A: PCA Linear, test window F1 `0.8571`.
- Family B: PCA Linear, test window F1 `0.9412`.

What the plots show:

Each plot overlays:

- Pink shading: true burst windows.
- Blue line: selected quantum model maximum IP-instance score per window.
- Green line: best online model maximum IP-instance score per window.
- Blue dashed threshold: selected quantum model calibrated threshold.
- Green dotted threshold: best online model threshold.
- Orange triangles: selected quantum model predicted attack windows.
- Purple diamonds: best online model predicted attack windows.

Current result interpretation:

| Family | Dataset | Quantum predicted | Quantum false positives | Quantum missed | Best online predicted | Best online false positives | Best online missed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| family_a | attack_test_01 | 15 | 11 | 0 | 5 | 1 | 0 |
| family_a | attack_test_02 | 15 | 10 | 0 | 6 | 1 | 0 |
| family_b | attack_test_01 | 13 | 10 | 0 | 3 | 0 | 0 |
| family_b | attack_test_02 | 13 | 8 | 0 | 6 | 1 | 0 |

Presentation takeaway:

> The overlay makes the false-positive story visually obvious. Quantum ZZ is highly recall-oriented and stays above threshold for many non-burst windows. PCA Linear, the strongest online baseline here, produces a much sharper temporal response: it still catches every burst in these test attack files, but it raises far fewer non-burst alerts.

How to present it:

Use one Family A overlay and one Family B overlay in the main deck. They make the comparison clearer than metric tables alone because the audience can see both timing and alert selectivity in the same figure.

## Training Curve Plot Family

Artifact family:

- `outputs/plots/training_curves_family_a.png`
- `outputs/plots/training_curves_family_b.png`

What they show:

Each training curve plot has two panels: train/validation loss and train/validation F1 over epochs.

How to read them:

- Falling loss with rising validation F1 indicates useful learning.
- A gap between train and validation F1 indicates overfitting.
- A flat validation F1 shows that the linear scorer has reached the limit of what the fixed quantum features can provide.
- Early stopping chooses the best validation-F1 checkpoint, not necessarily the final epoch.

Presentation takeaway:

> Training curves are model-development evidence. They show whether the small linear scorer converged and whether performance was limited by optimization or by the fixed representation.

## Full Metrics Summary Table

Artifacts:

- `outputs/tables/metrics_summary.csv`
- `outputs/tables/metrics_summary.md`

What it shows:

This is the complete flattened version of the JSON result files, excluding confusion-matrix counts. It includes validation and test metrics, all methods, and all evaluation levels.

How to use it:

Use this table for appendix material, not a main slide. It is useful when you need exact values for validation vs. test, or when checking whether a claim from a plot is backed by the raw numbers.

Presentation takeaway:

> This table is the audit trail for all summary plots.

## Test Method Rankings Table

Artifacts:

- `outputs/tables/test_method_rankings.csv`
- `outputs/tables/test_method_rankings.md`

What it shows:

This table ranks methods by test F1 separately for each family and evaluation level.

Key ranking results:

- Family A IP-instance ranking: PCA Linear first, Bag RBF second, Classical RBF third, Quantum ZZ fourth, Majority fifth.
- Family A window ranking: Bag RBF first, PCA Linear second, Classical RBF third, Quantum ZZ fourth, Majority fifth.
- Family B IP-instance ranking: Bag RBF first, PCA Linear second, Classical RBF third, Quantum ZZ fourth, Majority fifth.
- Family B window ranking: Bag RBF first, PCA Linear second, Classical RBF third, Quantum ZZ fourth, Majority fifth.

Interpretation:

The ranking table makes the main comparative result unambiguous: Quantum ZZ works, but it is not the top-performing method on these current datasets. The PCA Linear ablation is especially important because it is simpler and stronger in the online setting.

Presentation wording:

> The strongest online model in these experiments is PCA Linear, not Quantum ZZ. That is a useful scientific finding: it tells us the engineered feature space is already highly informative, and future quantum feature maps need to add value beyond PCA rather than simply add complexity.

## Confusion Matrix Summary Table

Artifacts:

- `outputs/tables/confusion_matrices_summary.csv`
- `outputs/tables/confusion_matrices_summary.md`

What it shows:

This table provides the raw true-negative, false-positive, false-negative, and true-positive counts for every method, split, family, and evaluation level.

How to use it:

Use it when you need to explain why precision or recall has a particular value. For example, Family A Quantum ZZ window recall is `1.000` because it has `9` true-positive windows and `0` false-negative windows; precision is only `0.300` because it also has `21` false-positive windows.

Presentation takeaway:

> Confusion counts reveal the practical burden of the model. F1 compresses performance into one number, but false positives tell us how many alerts an analyst would actually see.

## Detection Latency Summary Table

Artifacts:

- `outputs/tables/detection_latency_summary.csv`
- `outputs/tables/detection_latency_summary.md`

What it shows:

This is the tabular version of the latency plot, including detected burst counts and missed burst counts.

Interpretation:

Quantum ZZ, Classical RBF, and PCA Linear all report zero-window latency and no missed bursts in both families. Majority misses all bursts. Bag RBF should not be interpreted as an online latency model.

Presentation takeaway:

> Latency is not the weakness of the online detectors. The weakness is the precision of the alerts and IP attributions.

## Threshold Summary Table

Artifacts:

- `outputs/tables/threshold_summary.csv`
- `outputs/tables/threshold_summary.md`

What it shows:

This table records the calibrated Quantum ZZ thresholds and validation F1 scores.

Interpretation:

The threshold is part of the model deployment contract. Changing it will directly change the recall/precision balance. The current thresholds favor not missing burst windows, which explains the high recall and the false-positive burden.

Presentation takeaway:

> The threshold is an operational policy knob. A stricter threshold could reduce false positives, but may sacrifice the zero-miss window behavior.

## Recommended Main-Deck Story

Use this order for a clear presentation:

1. Problem and task: online DDoS detection with IP attribution.
2. Architecture slide: engineered window features -> PCA angles -> fixed ZZ map -> tiny linear scorer.
3. Executive summary: `presentation_quantum_executive_summary.png`.
4. Temporal behavior: one or two temporal plots to show online alerting.
5. Quantum confusion matrix grid: show zero missed attack windows and false-positive tradeoff.
6. F1 comparison: show that PCA Linear and classical baselines outperform Quantum ZZ.
7. Interpretation: the current quantum map works as a compact recall-oriented detector, but the ablation shows PCA features already carry most of the useful signal.
8. Next steps: improve calibration, tune thresholds for analyst workload, test richer quantum maps, and evaluate on stealthier/more distributed attacks.

## Final Defensible Conclusion

Presentation-ready conclusion:

> The implemented quantum-inspired detector successfully produces online window alerts and IP-level scores from the same instance-level model. It detects all test attack bursts with zero-window latency, but its current thresholding produces many false positives and its IP attribution is weaker than simpler classical baselines. The strongest lesson is that the engineered causal features are highly informative; future quantum feature maps must add discriminative value beyond the PCA-linear representation to justify their complexity.
