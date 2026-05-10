# Pipeline: Inputs, Outputs, and Shapes at Every Stage

```
=============================================================================
STAGE 0 — Raw Flow Data
=============================================================================
Input:   CSV file (one per dataset); 100,000 rows × 25 columns
         Columns: src_ip, dst_ip, src_port, dst_port, protocol,
                  outbound_byte_ratio, duration, packets_per_second,
                  bytes_per_second, inter_packet_arrival_mean,
                  inter_packet_arrival_std, total_packets, total_bytes,
                  packet_size_avg, packet_size_std,
                  [audit: row_in_window, is_seeded_ddos, burst_id,
                   burst_phase, scenario, split, dataset_id, source_dataset,
                   Label, Attack]
Output:  Pandas DataFrame, same shape; audit columns retained for indexing
         and label derivation but never passed to any model

=============================================================================
STAGE 1 — Window Partitioning
=============================================================================
Input:   DataFrame with `row_in_window` column; scalar W (window_width)
Output:  DataFrame with new column `window_id` = row_in_window // W
         Number of distinct window_ids per dataset: ceil(100000 / W)
         Recommended W = 6667 → 15 windows per dataset

=============================================================================
STAGE 2 — Instance Label Derivation (training and evaluation only)
=============================================================================
Input:   DataFrame with columns `src_ip`, `window_id`, `is_seeded_ddos`,
         `burst_id`
Output:  Instance table: one row per (src_ip, window_id) pair
         Columns added:
           y_instance  ∈ {0, 1}  = max(is_seeded_ddos) over the group
           y_window    ∈ {0, 1}  = 1 if any is_seeded_ddos == 1 in window_id
                                     (for evaluation only, not training)
         Shape: (n_instances, 2)  where n_instances = number of unique
                (src_ip, window_id) pairs in this dataset

=============================================================================
STAGE 3 — Feature Engineering
=============================================================================
Input:   DataFrame with all flow columns; grouped by (src_ip, window_id)
         Cross-IP correlation features require access to the full window group
Output:  Feature matrix F ∈ R^{n_instances × d}   where d = 33
         One row per (src_ip, window_id) instance
         Columns (in order):
           [Base - Volume]
             flow_count, total_packets_sum, total_bytes_sum
           [Base - Rate]
             pps_mean, pps_max, bps_mean, bps_max
           [Base - Duration]
             duration_mean, duration_max
           [Base - Packet size]
             packet_size_avg_mean, packet_size_avg_std
           [Base - Timing]
             inter_arrival_mean_mean, inter_arrival_std_mean
           [Base - Balance]
             outbound_ratio_mean, outbound_ratio_min
           [Base - Destination]
             unique_dst_ip, unique_dst_port, unique_protocols,
             dst_concentration, dst_port_concentration
           [Base - Protocol]
             tcp_share, udp_share, icmp_share
           [Base - Threshold ratios]
             small_packet_share, high_pps_share, low_outbound_share
           [Correlation]
             same_dst_ip_source_count, same_dst_port_source_count,
             dst_ip_global_flow_share, dst_ip_global_source_share,
             co_targeting_score, window_dst_entropy, window_port_entropy
         All values are float64; no categorical columns; no NaNs
         (packet_size_avg_std NaNs filled with 0.0)

=============================================================================
STAGE 4 — Log Scaling
=============================================================================
Input:   F ∈ R^{n_instances × d}
Output:  F_log ∈ R^{n_instances × d}    (same shape)
         Transformation: log(1 + x) applied to columns
           flow_count, total_packets_sum, total_bytes_sum,
           pps_max, bps_max, duration_max
         All other columns passed through unchanged
         No parameters; no fitting required

=============================================================================
STAGE 5 — Standard Scaling
=============================================================================
Input:   F_log ∈ R^{n_instances × d}
Output:  F_scaled ∈ R^{n_instances × d}   (zero mean, unit variance)
Fit on:  Training instances only (sklearn StandardScaler)
         Saved to: outputs/preprocessor_{family}.pkl  (as part of Preprocessor)
         Applied identically at validation, test, and inference time

=============================================================================
STAGE 6 — PCA Projection
=============================================================================
Input:   F_scaled ∈ R^{n_instances × d}
Output:  Z ∈ R^{n_instances × k}    where k = config.pca_components (e.g. 8)
Fit on:  Training instances only (sklearn PCA, n_components=k)
         Saved to: outputs/preprocessor_{family}.pkl
         Cumulative explained variance ratio logged; warning if < 0.85
         Applied identically at validation, test, and inference time

=============================================================================
STAGE 7 — Angle Rescaling
=============================================================================
Input:   Z ∈ R^{n_instances × k}
Output:  Θ ∈ R^{n_instances × k}    where each entry ∈ (-π, π)
         Transformation: θ_{i,j} = 2 · arctan(z_{i,j})
         No parameters; no fitting; applied identically everywhere

=============================================================================
STAGE 8 — ZZFeatureMap Quantum Circuit
=============================================================================
Input:   Θ ∈ R^{n_instances × k}    (passed as NumPy array, one instance at a
                                      time or in batches)
Circuit: k qubits, reps = config.circuit_reps (e.g. 2), linear entanglement
         Structure per rep:
           H⊗k → Rz(θ_0),...,Rz(θ_{k-1}) → ZZ(θ_0,θ_1),...,ZZ(θ_{k-2},θ_{k-1})
Output:  Q ∈ R^{n_instances × (2k-1)}   (CUDA float32 tensor)
         Columns:
           ⟨Z_0⟩, ⟨Z_1⟩, ..., ⟨Z_{k-1}⟩,
           ⟨Z_0 Z_1⟩, ⟨Z_1 Z_2⟩, ..., ⟨Z_{k-2} Z_{k-1}⟩
         All values ∈ [-1, +1] (expectation values of ±1-eigenvalue operators)
         Computed via exact statevector simulation (Qiskit Aer, statevector)
         No trainable parameters; circuit structure fixed by k and reps
         Cached to: outputs/quantum_features_{family}_{split}.pt

=============================================================================
STAGE 9 — Linear Scoring Layer
=============================================================================
Input:   Q ∈ R^{n_instances × (2k-1)}   (CUDA float32 tensor)
Output:  s ∈ R^{n_instances}            (CUDA float32 tensor; raw logits)
         p̂ ∈ R^{n_instances}            (CUDA float32 tensor; sigmoid of s)
         p̂_i = σ(s_i) = 1/(1+exp(-s_i)) ∈ (0, 1)
Layer:   nn.Linear(2*k - 1, 1, bias=True) on CUDA
         Weight: w ∈ R^{2k-1};  Bias: b ∈ R
         Total trainable parameters: 2k   (e.g. 16 for k=8)
Loss:    BCEWithLogitsLoss with pos_weight = n_neg / n_pos (training set)
         Applied at instance level: L = mean over batch of BCE(s_i, y_i)

=============================================================================
STAGE 10 — Inference and Threshold
=============================================================================
Input:   p̂ ∈ R^{n_instances}  (per-instance malicious probabilities)
         Instance metadata: (src_ip, window_id, dataset_name) for each row
         Threshold τ from outputs/threshold_{family}.json

Instance-level output:
         ŷ_instance_i = 1 if p̂_i ≥ τ else 0
         Malicious IP shortlist for window t: {src_ip_i : ŷ_instance_i = 1}

Window-level output (derived, no additional model):
         ŷ_window_t = 1 if max(p̂_i for i in window t) ≥ τ else 0

Dataset-level output (derived, no additional model):
         ŷ_dataset = 1 if any window in dataset is flagged else 0

=============================================================================
```
