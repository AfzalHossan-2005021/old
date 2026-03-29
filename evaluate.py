"""
stfa/evaluate.py
================
Performance metrics for spatial transcriptomics alignment.

Provides:
  - cell_type_matching_metric()    (original, unchanged for compatibility)
  - get_perf_metrics()             (original, unchanged for compatibility)
  - get_perf_metrics_enhanced()    (extended with 6 biologically motivated metrics)
  - get_coherent_perf_metrics()    (new metrics for sparse coherent registration)
"""

from __future__ import annotations

import warnings
import numpy as np
from anndata import AnnData
from typing import List, Optional
from scipy.spatial.distance import cdist


# ──────────────────────────────────────────────────────────────────────────────
# Original metrics (API-compatible)
# ──────────────────────────────────────────────────────────────────────────────

def cell_type_matching_metric(sliceA: AnnData, sliceB: AnnData, pi_mat: np.ndarray) -> float:
    """
    Expected fraction (%) of transport mass that lands on the correct cell type.
    Probabilistic: works for soft/unbalanced transport plans.
    """
    pi_arr = np.asarray(pi_mat, dtype=np.float64)
    expected_matches = 0.0
    for i in range(sliceA.n_obs):
        src_type = sliceA.obs.iloc[i]['cell_type_annot']
        match_mask = np.asarray(sliceB.obs['cell_type_annot'].values == src_type, dtype=bool)
        expected_matches += float(np.sum(pi_arr[i, match_mask]))
    total_mass = float(pi_arr.sum())
    if total_mass > 0:
        return (expected_matches / total_mass) * 100.0
    return 0.0


def get_perf_metrics(
    new_slices: List[AnnData],
    pi12: np.ndarray,
    neighbor_initial_obj: float,
    initial_obj_gene_cos: float,
    neighbor_final_obj: float,
    obj_gene_cos: float,
    method_name: str,
) -> None:
    """Original three-metric report. API-compatible with old code."""
    neighborhood_improvement = (
        (neighbor_initial_obj - neighbor_final_obj) / (neighbor_initial_obj + 1e-12) * 100
    )
    gene_expr_improvement = (
        (initial_obj_gene_cos - obj_gene_cos) / (initial_obj_gene_cos + 1e-12) * 100
    )
    expected_matches = cell_type_matching_metric(new_slices[0], new_slices[1], pi12)

    print(f"\n{'─'*55}")
    print(f"  Method: {method_name}")
    print(f"{'─'*55}")
    print(f"  JSD Neighbourhood")
    print(f"    Before: {neighbor_initial_obj:.7f}  After: {neighbor_final_obj:.7f}  "
          f"Δ: {neighborhood_improvement:+.4f}%")
    print(f"  Cosine Gene Expression")
    print(f"    Before: {initial_obj_gene_cos:.7f}  After: {obj_gene_cos:.7f}  "
          f"Δ: {gene_expr_improvement:+.4f}%")
    print(f"  Cell-type Correspondence: {expected_matches:.4f}%")
    print(f"{'─'*55}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Enhanced metrics
# ──────────────────────────────────────────────────────────────────────────────

def get_perf_metrics_enhanced(
    new_slices: List[AnnData],
    pi12: np.ndarray,
    neighbor_initial_obj: float,
    initial_obj_gene_cos: float,
    neighbor_final_obj: float,
    obj_gene_cos: float,
    method_name: str,
    k_nn: int = 15,
) -> dict:
    """
    Extended performance report with 7 biologically motivated additions:

    1.  Unmatched mass fraction  D_global      — tissue overlap quality
    2.  Spatial residual RMSE                  — geometric alignment error
    3.  k-NN Neighbourhood Structure Pres. NSP — local topology conservation
    4.  Transport entropy                       — symmetry confidence
    5.  Rare cell-type recall (< 1% freq)      — cross-temporal sensitivity
    6.  Symmetry ambiguity score               — whether top-2 targets compete
    7.  Mapped-region geometry consistency      — internal shape preservation

    Returns a dict of all metric values (for programmatic use).
    """
    from sklearn.neighbors import NearestNeighbors

    pi_arr = np.asarray(pi12, dtype=np.float64)

    def _weighted_rigid_fit(src: np.ndarray, tgt: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w = np.asarray(w, dtype=np.float64)
        mass = float(w.sum())
        if mass <= 0:
            return np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)

        ws = w / (mass + 1e-12)
        c_src = (ws[:, None] * src).sum(axis=0)
        c_tgt = (ws[:, None] * tgt).sum(axis=0)
        src_c = src - c_src
        tgt_c = tgt - c_tgt

        H = (src_c * ws[:, None]).T @ tgt_c
        U, _, Vt = np.linalg.svd(H, full_matrices=False)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T

        return R.astype(np.float64), c_src.astype(np.float64), c_tgt.astype(np.float64)

    def _norm_pairwise(D: np.ndarray) -> np.ndarray:
        tri = D[np.triu_indices_from(D, k=1)]
        tri = tri[np.isfinite(tri)]
        tri = tri[tri > 0]
        if tri.size == 0:
            return D
        scale = float(np.median(tri))
        if scale <= 0:
            return D
        return D / (scale + 1e-12)

    def _mapped_region_geometry(pi: np.ndarray, src_xy: np.ndarray, tgt_xy: np.ndarray) -> tuple[float, float]:
        row_mass_loc = pi.sum(axis=1)
        valid = row_mass_loc > 1e-12
        if int(valid.sum()) < 5:
            return float('nan'), float('nan')

        # Focus on cells that actually carry transport mass (mapped region).
        nz = row_mass_loc[valid]
        q = float(np.quantile(nz, 0.35))
        keep = row_mass_loc >= max(1e-12, q)
        if int(keep.sum()) < 5:
            keep = valid

        row_safe = row_mass_loc[:, None] + 1e-12
        bary = (pi / row_safe) @ tgt_xy

        src_sel = np.asarray(src_xy[keep], dtype=np.float64)
        bary_sel = np.asarray(bary[keep], dtype=np.float64)
        w_sel = np.asarray(row_mass_loc[keep], dtype=np.float64)
        if src_sel.shape[0] < 5:
            return float('nan'), float('nan')

        R, c_src, c_tgt = _weighted_rigid_fit(src_sel, bary_sel, w_sel)
        src_aligned = (R @ (src_sel - c_src).T).T + c_tgt

        D_src = _norm_pairwise(cdist(src_aligned, src_aligned))
        D_tgt = _norm_pairwise(cdist(bary_sel, bary_sel))

        tri_idx = np.triu_indices(src_sel.shape[0], k=1)
        x = D_src[tri_idx]
        y = D_tgt[tri_idx]

        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size < 5:
            return float('nan'), float('nan')

        x_std = float(np.std(x))
        y_std = float(np.std(y))
        if x_std <= 1e-12 or y_std <= 1e-12:
            corr = float('nan')
        else:
            corr = float(np.corrcoef(x, y)[0, 1])

        stress = float(np.sqrt(np.sum((x - y) ** 2) / (np.sum(x ** 2) + 1e-12)))
        return corr, stress

    N_A, N_B = pi_arr.shape
    p_A = np.ones(N_A) / N_A

    # ── Original three ──────────────────────────────────────────────────────
    neighborhood_improvement = (
        (neighbor_initial_obj - neighbor_final_obj) / (neighbor_initial_obj + 1e-12) * 100
    )
    gene_expr_improvement = (
        (initial_obj_gene_cos - obj_gene_cos) / (initial_obj_gene_cos + 1e-12) * 100
    )
    expected_matches = cell_type_matching_metric(new_slices[0], new_slices[1], pi_arr)

    # ── 1. Unmatched mass fraction ────────────────────────────────────────────
    # Under KL-unbalanced OT the plan does NOT sum to 1/N_A per row.
    # D_global = fraction of source mass not transported anywhere.
    row_mass = pi_arr.sum(axis=1)                  # (N_A,)
    D_local  = np.clip(p_A - row_mass, 0, None)  # (N_A,) per-cell destroyed mass
    D_global = float(D_local.sum())               # fraction ∈ [0, 1]

    # ── 2. Spatial residual RMSE ──────────────────────────────────────────────
    s1 = np.asarray(new_slices[0].obsm['spatial'], dtype=np.float64)
    s2 = np.asarray(new_slices[1].obsm['spatial'], dtype=np.float64)
    pi_row_safe = row_mass[:, None] + 1e-12
    # Weighted average target position for each source cell
    src_mapped = (pi_arr / pi_row_safe) @ s2          # (N_A, 2)
    residuals  = np.linalg.norm(s1 - src_mapped, axis=1)
    # Weight RMSE by how much mass each cell actually transported
    match_weight = row_mass * N_A                    # ~1 for matched, ~0 for unmatched
    spatial_rmse = float(
        np.sqrt(np.average(residuals ** 2, weights=match_weight + 1e-12))
    )

    # ── 3. k-NN Neighbourhood Structure Preservation (NSP) ───────────────────
    nn_s1 = NearestNeighbors(n_neighbors=k_nn).fit(s1)
    nn_s2 = NearestNeighbors(n_neighbors=k_nn).fit(s2)
    _, idx_s1 = nn_s1.kneighbors(s1)
    _, idx_s2 = nn_s2.kneighbors(s2)
    best_match = np.argmax(pi_arr, axis=1)          # greedy 1-1 for NSP

    nsp_scores = []
    for i in range(N_A):
        j = best_match[i]
        mapped_nbrs = set(best_match[idx_s1[i]])
        true_nbrs   = set(idx_s2[j])
        jaccard = len(mapped_nbrs & true_nbrs) / (len(mapped_nbrs | true_nbrs) + 1e-12)
        nsp_scores.append(jaccard)
    nsp = float(np.mean(nsp_scores) * 100)

    # ── 4. Transport entropy ─────────────────────────────────────────────────
    pi_norm    = pi_arr / (pi_arr.sum() + 1e-12)
    entropy    = float(-np.sum(pi_norm * np.log(pi_norm + 1e-12)))
    max_entropy = float(np.log(N_A * N_B))
    entropy_pct = entropy / max_entropy * 100

    # ── 4b. Effective support (confidence) in both directions ───────────────
    # exp(H) is the entropy-equivalent number of active correspondences.
    # Lower is tighter/more confident.
    row_mass_safe = row_mass[:, None] + 1e-12
    col_mass_safe = pi_arr.sum(axis=0, keepdims=True) + 1e-12
    row_cond = pi_arr / row_mass_safe
    col_cond = pi_arr / col_mass_safe
    row_entropy = -np.sum(row_cond * np.log(row_cond + 1e-12), axis=1)
    col_entropy = -np.sum(col_cond * np.log(col_cond + 1e-12), axis=0)
    eff_targets_per_source = float(np.mean(np.exp(row_entropy)))
    eff_sources_per_target = float(np.mean(np.exp(col_entropy)))

    # ── 5. Rare cell-type recall ──────────────────────────────────────────────
    all_types  = new_slices[0].obs['cell_type_annot']
    type_freq  = all_types.value_counts(normalize=True)
    rare_types = type_freq[type_freq < 0.01].index.tolist()

    if rare_types:
        rare_mask = new_slices[0].obs['cell_type_annot'].isin(rare_types).values
        rare_mask = np.asarray(rare_mask, dtype=bool)
        pi_rare   = pi_arr[rare_mask, :]
        rare_recall = cell_type_matching_metric(
            new_slices[0][rare_mask], new_slices[1], pi_rare
        )
        rare_str = f"{rare_recall:.4f}%"
    else:
        rare_recall = float('nan')
        rare_str = "N/A (no cell types < 1%)"

    # ── 6. Symmetry ambiguity score ───────────────────────────────────────────
    sorted_rows = np.sort(pi_arr, axis=1)          # ascending
    top1  = sorted_rows[:, -1] + 1e-12
    top2  = sorted_rows[:, -2] + 1e-12
    sym_ambig = float(np.mean(top2 / top1))      # 1.0 = fully ambiguous

    # ── 7. Mapped-region internal geometry consistency ───────────────────────
    mapped_geom_corr, mapped_geom_stress = _mapped_region_geometry(pi_arr, s1, s2)

    # ── Report ────────────────────────────────────────────────────────────────
    sep = '═' * 60
    print(f"\n{sep}")
    print(f"  {method_name} — Enhanced Performance Report")
    print(sep)
    # Original
    print(f"  JSD Neighbourhood     "
          f"Before: {neighbor_initial_obj:.6f}  "
          f"After: {neighbor_final_obj:.6f}  "
          f"Δ: {neighborhood_improvement:+.3f}%")
    print(f"  Cosine Gene Expr      "
          f"Before: {initial_obj_gene_cos:.6f}  "
          f"After: {obj_gene_cos:.6f}  "
          f"Δ: {gene_expr_improvement:+.3f}%")
    print(f"  Cell-Type Match       {expected_matches:.4f}%")
    # New
    print(f"{'─'*60}")
    print(f"  Unmatched Mass (D_global)      {D_global*100:.2f}% of source")
    print(f"  Spatial RMSE                   {spatial_rmse:.4f} (coord units)")
    print(f"  k-NN NSP (k={k_nn})             {nsp:.4f}%")
    print(f"  Transport Entropy              {entropy_pct:.2f}% of max "
        f"(low = confident)")
    print(f"  Effective Targets / Source     {eff_targets_per_source:.3f} "
        f"(low = tighter)")
    print(f"  Effective Sources / Target     {eff_sources_per_target:.3f} "
        f"(low = tighter)")
    print(f"  Rare Cell-Type Recall          {rare_str}")
    print(f"  Symmetry Ambiguity Score       {sym_ambig:.4f} "
          f"(1.0 = fully ambiguous)")
    if np.isfinite(mapped_geom_corr):
        print(f"  Mapped Geometry Corr (R)       {mapped_geom_corr:.4f} "
              f"(high = better)")
    else:
        print("  Mapped Geometry Corr (R)       N/A")
    if np.isfinite(mapped_geom_stress):
        print(f"  Mapped Geometry Stress         {mapped_geom_stress:.4f} "
              f"(low = better)")
    else:
        print("  Mapped Geometry Stress         N/A")
    print(f"{sep}\n")

    return {
        'jsd_before':         neighbor_initial_obj,
        'jsd_after':          neighbor_final_obj,
        'jsd_improvement_pct': neighborhood_improvement,
        'cosine_before':      initial_obj_gene_cos,
        'cosine_after':       obj_gene_cos,
        'cosine_improvement_pct': gene_expr_improvement,
        'cell_type_match_pct':    expected_matches,
        'unmatched_mass_pct': D_global * 100,
        'spatial_rmse':       spatial_rmse,
        'nsp_pct':            nsp,
        'entropy_pct':        entropy_pct,
        'eff_targets_per_source': eff_targets_per_source,
        'eff_sources_per_target': eff_sources_per_target,
        'rare_cell_recall_pct': rare_recall,
        'symmetry_ambiguity': sym_ambig,
        'mapped_geom_corr': mapped_geom_corr,
        'mapped_geom_stress': mapped_geom_stress,
    }


def get_coherent_perf_metrics(
    sliceA: AnnData,
    sliceB: AnnData,
    result,
    k_nn: int = 15,
    verbose: bool = True,
) -> dict:
    """
    Metrics for the new coherent partial registration pipeline.

    This uses transformed coordinates, sparse transport plans, and explicit
    compactness/cycle diagnostics, which makes it valid under arbitrary rigid
    transforms and avoids the dummy-mass renormalization issue in the baseline
    enhanced metrics.
    """
    from .coherent import summarize_alignment_metrics

    metrics = summarize_alignment_metrics(sliceA, sliceB, result, k_nn=k_nn)
    if verbose:
        sep = "=" * 60
        print(f"\n{sep}")
        print("  Coherent Partial Registration Report")
        print(sep)
        print(f"  Cell-Type Match              {metrics['cell_type_match_pct']:.4f}%")
        print(f"  Matched Mass                 {metrics['matched_mass_pct']:.2f}%")
        print(f"  Unmatched Source Mass        {metrics['unmatched_src_mass_pct']:.2f}%")
        print(f"  Unmatched Target Mass        {metrics['unmatched_tgt_mass_pct']:.2f}%")
        print(f"  Spatial RMSE                 {metrics['spatial_rmse']:.4f}")
        print(f"  Spatial RMSE (normalized)    {metrics.get('spatial_rmse_norm', float('nan')):.4f}")
        print(f"  k-NN NSP (k={k_nn})           {metrics['nsp_pct']:.4f}%")
        print(f"  Entropy                      {metrics['entropy_pct']:.2f}% of sparse support")
        print(f"  Effective Targets/Source     {metrics['eff_targets_per_source']:.3f}")
        print(f"  Effective Sources/Target     {metrics['eff_sources_per_target']:.3f}")
        print(f"  Symmetry Ambiguity           {metrics['symmetry_ambiguity']:.4f}")
        print(f"  Forward Compactness          {metrics['forward_compactness']:.4f}")
        print(f"  Reverse Compactness          {metrics['reverse_compactness']:.4f}")
        print(f"  Cycle Error                  {metrics['cycle_error']:.4f}")
        print(f"  Cycle Error (normalized)     {metrics.get('cycle_error_norm', float('nan')):.4f}")
        print(f"  Mapped Geometry Corr         {metrics['mapped_geom_corr']:.4f}")
        print(f"  Mapped Geometry Stress       {metrics['mapped_geom_stress']:.4f}")
        print(f"  Selection Score              {metrics['selection_score']:.4f}")
        print(f"{sep}\n")
    return metrics
