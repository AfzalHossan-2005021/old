from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Iterable, Literal, Optional

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from scipy.spatial.distance import cdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors


ArrayLike = np.ndarray


@dataclass
class RigidTransform:
    matrix: ArrayLike
    translation: ArrayLike
    reflected: bool = False

    def apply(self, points: ArrayLike) -> ArrayLike:
        pts = np.asarray(points, dtype=np.float64)
        return pts @ self.matrix.T + self.translation


@dataclass
class DeformationField:
    control_points: ArrayLike
    weights: ArrayLike
    gamma: float

    def apply(self, points: ArrayLike) -> ArrayLike:
        pts = np.asarray(points, dtype=np.float64)
        if self.control_points.size == 0 or self.weights.size == 0:
            return pts.copy()
        diff = pts[:, None, :] - self.control_points[None, :, :]
        sqdist = np.sum(diff * diff, axis=2)
        kernel = np.exp(-self.gamma * sqdist)
        return pts + kernel @ self.weights


@dataclass
class AlignmentResult:
    pi_fwd: sp.csr_matrix
    pi_rev: sp.csr_matrix
    raw_unmatched_src: ArrayLike
    raw_unmatched_tgt: ArrayLike
    selected_hypothesis: Dict[str, Any]
    hypothesis_scores: list[Dict[str, Any]]
    rigid_transform: RigidTransform
    deformation_field: Optional[DeformationField]
    src_warped: ArrayLike
    tgt_warped: ArrayLike
    forward_barycenters: ArrayLike
    reverse_barycenters: ArrayLike
    metrics: Dict[str, float]


def coherent_pairwise_align(
    sliceA: AnnData,
    sliceB: AnnData,
    mode: Literal["same_timepoint", "cross_timepoint"] = "same_timepoint",
    use_rep: Optional[str] = None,
    max_hypotheses: int = 8,
    candidate_k: int = 8,
    n_iters: int = 5,
    n_supercells: Optional[int] = None,
    neighborhood_radius: Optional[float] = None,
    random_state: int = 0,
    verbose: bool = False,
) -> AlignmentResult:
    if mode not in {"same_timepoint", "cross_timepoint"}:
        raise ValueError("mode must be 'same_timepoint' or 'cross_timepoint'.")

    rng = np.random.default_rng(random_state)
    procA, procB = _prepare_slices(sliceA, sliceB)

    coords_A = np.asarray(procA.obsm["spatial"], dtype=np.float64)
    coords_B = np.asarray(procB.obsm["spatial"], dtype=np.float64)
    expr_A = _get_matrix(procA, use_rep)
    expr_B = _get_matrix(procB, use_rep)
    labels_A = np.asarray(procA.obs["cell_type_annot"].astype(str).values)
    labels_B = np.asarray(procB.obs["cell_type_annot"].astype(str).values)

    expr_A_n = _row_normalize(expr_A + 1e-8)
    expr_B_n = _row_normalize(expr_B + 1e-8)

    scale_A, idx_A, dist_A = _adaptive_scale(coords_A)
    scale_B, idx_B, dist_B = _adaptive_scale(coords_B)
    spatial_scale = float(max(np.median(dist_A[:, -1]), np.median(dist_B[:, -1]), 1e-6))
    if neighborhood_radius is None:
        neighborhood_radius = 1.75 * 0.5 * (scale_A + scale_B)

    graph_A = _knn_graph(coords_A, idx_A, dist_A, scale_A)
    graph_B = _knn_graph(coords_B, idx_B, dist_B, scale_B)

    union_types = sorted(set(labels_A.tolist()) | set(labels_B.tolist()))
    nbhd_A = _neighborhood_profiles(coords_A, labels_A, union_types, neighborhood_radius)
    nbhd_B = _neighborhood_profiles(coords_B, labels_B, union_types, neighborhood_radius)

    diff_A = _diffusion_signatures(graph_A)
    diff_B = _diffusion_signatures(graph_B)
    shape_A = _local_shape_context_descriptors(coords_A)
    shape_B = _local_shape_context_descriptors(coords_B)
    boundary_A = _boundary_confidence(dist_A[:, -1])
    boundary_B = _boundary_confidence(dist_B[:, -1])

    if n_supercells is None:
        n_supercells = int(max(16, min(256, np.sqrt(max(procA.n_obs, procB.n_obs)))))

    comm_A = _supercell_labels(coords_A, n_supercells, rng)
    comm_B = _supercell_labels(coords_B, n_supercells, rng)
    coarse_A = _aggregate_supercells(expr_A_n, diff_A, shape_A, labels_A, union_types, boundary_A, coords_A, comm_A)
    coarse_B = _aggregate_supercells(expr_B_n, diff_B, shape_B, labels_B, union_types, boundary_B, coords_B, comm_B)

    hypotheses = _enumerate_hypotheses(
        coarse_A["centroids"],
        coarse_B["centroids"],
        coarse_A["descriptors"],
        coarse_B["descriptors"],
        max_hypotheses=max_hypotheses,
    )
    if not hypotheses:
        identity = RigidTransform(matrix=np.eye(2), translation=np.zeros(2), reflected=False)
        hypotheses = [{"transform": identity, "score": np.inf, "source": "fallback"}]

    type_fallback_A2B = _type_fallback_map(expr_A_n, expr_B_n, labels_A, labels_B)
    type_fallback_B2A = _type_fallback_map(expr_B_n, expr_A_n, labels_B, labels_A)

    best_result: Optional[AlignmentResult] = None
    hypothesis_scores: list[Dict[str, Any]] = []

    for h_idx, hypothesis in enumerate(hypotheses):
        result = _refine_hypothesis(
            sliceA=procA,
            sliceB=procB,
            expr_A_n=expr_A_n,
            expr_B_n=expr_B_n,
            nbhd_A=nbhd_A,
            nbhd_B=nbhd_B,
            coords_A=coords_A,
            coords_B=coords_B,
            labels_A=labels_A,
            labels_B=labels_B,
            graph_A=graph_A,
            graph_B=graph_B,
            neighbor_idx_A=idx_A,
            neighbor_idx_B=idx_B,
            spatial_scale=spatial_scale,
            boundary_A=boundary_A,
            boundary_B=boundary_B,
            type_fallback_A2B=type_fallback_A2B,
            type_fallback_B2A=type_fallback_B2A,
            mode=mode,
            init_transform=hypothesis["transform"],
            candidate_k=candidate_k,
            n_iters=n_iters,
            verbose=verbose,
        )
        result.selected_hypothesis = {
            "index": h_idx,
            "source": hypothesis.get("source", "coarse"),
            "initial_score": float(hypothesis["score"]),
            "reflected": bool(result.rigid_transform.reflected),
        }
        hypothesis_scores.append(
            {
                "index": h_idx,
                "source": hypothesis.get("source", "coarse"),
                "initial_score": float(hypothesis["score"]),
                "final_score": float(result.metrics["selection_score"]),
                "reflected": bool(result.rigid_transform.reflected),
            }
        )
        if best_result is None or result.metrics["selection_score"] < best_result.metrics["selection_score"]:
            best_result = result

    assert best_result is not None
    best_result.hypothesis_scores = hypothesis_scores
    best_result.selected_hypothesis["final_score"] = float(best_result.metrics["selection_score"])
    return best_result


def summarize_alignment_metrics(
    sliceA: AnnData,
    sliceB: AnnData,
    result: AlignmentResult,
    k_nn: int = 15,
    max_geom_points: int = 512,
) -> Dict[str, float]:
    coords_B = np.asarray(sliceB.obsm["spatial"], dtype=np.float64)
    src_warped = np.asarray(result.src_warped, dtype=np.float64)

    pi_fwd = result.pi_fwd.tocsr()
    pi_rev = result.pi_rev.tocsr()
    row_mass = np.asarray(pi_fwd.sum(axis=1)).ravel()

    src_types = np.asarray(sliceA.obs["cell_type_annot"].astype(str).values)
    tgt_types = np.asarray(sliceB.obs["cell_type_annot"].astype(str).values)

    cell_type_match = _sparse_type_match(pi_fwd, src_types, tgt_types)
    entropy_pct, eff_targets = _sparse_entropy_support(pi_fwd)
    _, eff_sources = _sparse_entropy_support(pi_rev)
    symmetry_ambiguity = _sparse_symmetry_ambiguity(pi_fwd)
    best_targets = _sparse_best_match(pi_fwd)
    valid_best = best_targets >= 0

    spatial_rmse = _safe_weighted_rmse(src_warped, result.forward_barycenters, row_mass)
    nsp = _nsp_score(src_warped, coords_B, best_targets, valid_best, k_nn=k_nn)
    mapped_corr, mapped_stress = _mapped_geometry_consistency(
        src_warped,
        result.forward_barycenters,
        row_mass,
        max_points=max_geom_points,
    )
    forward_compactness = _compactness_score(pi_fwd, coords_B, result.forward_barycenters)
    reverse_compactness = _compactness_score(pi_rev, src_warped, result.reverse_barycenters)
    cycle_error = _cycle_error(pi_fwd, result.reverse_barycenters, src_warped)

    metrics = {
        "cell_type_match_pct": cell_type_match,
        "matched_mass_pct": float(pi_fwd.sum() * 100.0),
        "unmatched_src_mass_pct": float(result.raw_unmatched_src.sum() * 100.0),
        "unmatched_tgt_mass_pct": float(result.raw_unmatched_tgt.sum() * 100.0),
        "spatial_rmse": spatial_rmse,
        "nsp_pct": nsp,
        "entropy_pct": entropy_pct,
        "eff_targets_per_source": eff_targets,
        "eff_sources_per_target": eff_sources,
        "symmetry_ambiguity": symmetry_ambiguity,
        "mapped_geom_corr": mapped_corr,
        "mapped_geom_stress": mapped_stress,
        "forward_compactness": forward_compactness,
        "reverse_compactness": reverse_compactness,
        "cycle_error": cycle_error,
    }
    metrics["selection_score"] = _selection_score(metrics)
    return metrics


def _prepare_slices(sliceA: AnnData, sliceB: AnnData) -> tuple[AnnData, AnnData]:
    if sliceA.n_obs == 0 or sliceB.n_obs == 0:
        raise ValueError("Both slices must contain at least one cell.")
    shared_genes = sliceA.var_names.intersection(sliceB.var_names)
    if len(shared_genes) == 0:
        raise ValueError("No shared genes between the two slices.")
    procA = sliceA[:, shared_genes].copy()
    procB = sliceB[:, shared_genes].copy()
    if "spatial" not in procA.obsm or "spatial" not in procB.obsm:
        raise ValueError("Both slices must contain obsm['spatial'].")
    if "cell_type_annot" not in procA.obs or "cell_type_annot" not in procB.obs:
        raise ValueError("Both slices must contain obs['cell_type_annot'].")
    return procA, procB


def _get_matrix(adata: AnnData, use_rep: Optional[str]) -> ArrayLike:
    X = adata.X if use_rep is None else adata.obsm[use_rep]
    if sp.issparse(X):
        return X.toarray().astype(np.float64)
    return np.asarray(X, dtype=np.float64)


def _row_normalize(X: ArrayLike) -> ArrayLike:
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms


def _adaptive_scale(coords: ArrayLike, k: int = 10) -> tuple[float, ArrayLike, ArrayLike]:
    n = coords.shape[0]
    k_eff = int(max(2, min(k + 1, n)))
    nn = NearestNeighbors(n_neighbors=k_eff)
    nn.fit(coords)
    dist, idx = nn.kneighbors(coords)
    if k_eff <= 1:
        return 1.0, idx[:, :0], dist[:, :0]
    scale = float(np.median(dist[:, -1]))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return scale, idx[:, 1:], dist[:, 1:]


def _knn_graph(coords: ArrayLike, idx: ArrayLike, dist: ArrayLike, scale: float) -> sp.csr_matrix:
    if idx.size == 0:
        return sp.csr_matrix((coords.shape[0], coords.shape[0]), dtype=np.float64)
    rows = np.repeat(np.arange(coords.shape[0]), idx.shape[1])
    cols = idx.ravel()
    weights = np.exp(-(dist.ravel() ** 2) / (2.0 * (scale ** 2 + 1e-12)))
    graph = sp.csr_matrix((weights, (rows, cols)), shape=(coords.shape[0], coords.shape[0]))
    graph = 0.5 * (graph + graph.T)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph.tocsr()


def _boundary_confidence(kth_dist: ArrayLike) -> ArrayLike:
    kth = np.asarray(kth_dist, dtype=np.float64)
    q_lo, q_hi = np.percentile(kth, [10, 90])
    scale = max(q_hi - q_lo, 1e-6)
    z = np.clip((kth - q_lo) / scale, 0.0, 1.0)
    return 1.0 - 0.8 * z


def _neighborhood_profiles(
    coords: ArrayLike,
    labels: ArrayLike,
    union_types: list[str],
    radius: float,
) -> ArrayLike:
    label_to_idx = {label: i for i, label in enumerate(union_types)}
    encoded = np.array([label_to_idx[label] for label in labels], dtype=np.int64)
    nn = NearestNeighbors(radius=radius)
    nn.fit(coords)
    neighbors = nn.radius_neighbors(coords, return_distance=False)
    out = np.zeros((coords.shape[0], len(union_types)), dtype=np.float64)
    for i, nbrs in enumerate(neighbors):
        if nbrs.size == 0:
            out[i, encoded[i]] = 1.0
            continue
        vals, counts = np.unique(encoded[nbrs], return_counts=True)
        out[i, vals] = counts
    out += 1e-6
    return out / out.sum(axis=1, keepdims=True)


def _diffusion_signatures(graph: sp.csr_matrix, n_components: int = 8) -> ArrayLike:
    n = graph.shape[0]
    if n == 0:
        return np.zeros((0, n_components), dtype=np.float64)
    if n <= n_components:
        dense = graph.toarray().astype(np.float64)
        if dense.shape[1] < n_components:
            dense = np.pad(dense, ((0, 0), (0, n_components - dense.shape[1])))
        return _row_normalize(dense[:, :n_components] + 1e-8)
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    emb = svd.fit_transform(graph)
    return _row_normalize(emb + 1e-8)


def _local_shape_context_descriptors(
    coords: ArrayLike,
    k_neighbors: int = 24,
    n_radial_bins: int = 6,
    n_angular_bins: int = 12,
    n_fourier: int = 4,
) -> ArrayLike:
    pts = np.asarray(coords, dtype=np.float64)
    n = int(pts.shape[0])
    n_r = int(max(3, n_radial_bins))
    n_a = int(max(8, n_angular_bins))
    n_f = int(max(2, n_fourier))
    d_dim = n_r + n_f + 3

    if n == 0:
        return np.zeros((0, d_dim), dtype=np.float64)
    if n == 1:
        return np.zeros((1, d_dim), dtype=np.float64)

    k = int(max(1, min(k_neighbors, n - 1)))
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(pts)
    _, nbr_idx = nn.kneighbors(pts, return_distance=True)
    nbr_idx = nbr_idx[:, 1:]

    radial_edges = np.geomspace(0.25, 4.0, num=n_r + 1)
    angular_edges = np.linspace(-np.pi, np.pi, num=n_a + 1)
    out = np.zeros((n, d_dim), dtype=np.float64)

    for i in range(n):
        neighbors = pts[nbr_idx[i]]
        vec = neighbors - pts[i]
        dist = np.linalg.norm(vec, axis=1) + 1e-12
        local_scale = np.median(dist) + 1e-12
        dist_n = dist / local_scale

        h_r, _ = np.histogram(dist_n, bins=radial_edges)
        h_r = h_r.astype(np.float64)
        h_r /= h_r.sum() + 1e-12

        ang = np.arctan2(vec[:, 1], vec[:, 0])
        h_a, _ = np.histogram(ang, bins=angular_edges)
        h_a = h_a.astype(np.float64)
        fft_mag = np.abs(np.fft.rfft(h_a))
        ang_inv = fft_mag[1:1 + n_f]
        if ang_inv.size < n_f:
            ang_inv = np.pad(ang_inv, (0, n_f - ang_inv.size))
        ang_inv = ang_inv / (h_a.sum() + 1e-12)

        if vec.shape[0] >= 2:
            cov = np.cov((vec / local_scale).T)
        else:
            cov = np.eye(2, dtype=np.float64)
        eig = np.linalg.eigvalsh(cov)
        eig = np.clip(np.sort(eig), 0.0, None)
        anis = float((eig[-1] - eig[0]) / (eig[-1] + eig[0] + 1e-12))

        out[i] = np.concatenate([h_r, ang_inv, [float(np.mean(dist_n)), float(np.std(dist_n)), anis]])

    return _row_normalize(out + 1e-8)


def _supercell_labels(coords: ArrayLike, n_clusters: int, rng: np.random.Generator) -> ArrayLike:
    n = coords.shape[0]
    k = int(max(1, min(n_clusters, n)))
    if k == 1:
        return np.zeros(n, dtype=np.int64)
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=int(rng.integers(0, 2**31 - 1)),
        batch_size=min(2048, n),
        n_init=5,
    )
    return model.fit_predict(coords)


def _aggregate_supercells(
    expr: ArrayLike,
    diff: ArrayLike,
    shape: ArrayLike,
    labels: ArrayLike,
    union_types: list[str],
    boundary: ArrayLike,
    coords: ArrayLike,
    comm: ArrayLike,
) -> Dict[str, ArrayLike]:
    uniq = np.unique(comm)
    type_to_idx = {t: i for i, t in enumerate(union_types)}
    descs = []
    cents = []

    for c in uniq:
        mask = comm == c
        if not np.any(mask):
            continue
        gene = expr[mask].mean(axis=0)
        topo = diff[mask].mean(axis=0)
        shp = shape[mask].mean(axis=0)
        ct = np.zeros(len(union_types), dtype=np.float64)
        for lbl in labels[mask]:
            ct[type_to_idx[lbl]] += 1.0
        ct /= ct.sum() + 1e-12
        aux = np.array([float(mask.mean()), float(boundary[mask].mean())], dtype=np.float64)
        desc = np.concatenate([gene, topo, shp, ct, aux])
        descs.append(desc)
        cents.append(coords[mask].mean(axis=0))

    desc_arr = _row_normalize(np.vstack(descs) + 1e-8)
    cent_arr = np.vstack(cents).astype(np.float64)
    return {"descriptors": desc_arr, "centroids": cent_arr}


def _enumerate_hypotheses(
    cent_A: ArrayLike,
    cent_B: ArrayLike,
    desc_A: ArrayLike,
    desc_B: ArrayLike,
    max_hypotheses: int,
) -> list[Dict[str, Any]]:
    M = cosine_distances(desc_A, desc_B)
    sort_idx = np.argsort(M, axis=1)
    top1 = M[np.arange(M.shape[0]), sort_idx[:, 0]]
    top2 = M[np.arange(M.shape[0]), sort_idx[:, 1]] if M.shape[1] > 1 else np.ones(M.shape[0])
    gaps = top2 - top1
    anchor_pool = np.argsort(-gaps)[: min(6, M.shape[0])]

    hypotheses: list[Dict[str, Any]] = []
    sim = np.exp(-M / (np.median(M) + 1e-6))
    sim /= sim.sum() + 1e-12

    for allow_reflection in (False, True):
        transform = _fit_transform_from_coupling(cent_A, cent_B, sim, allow_reflection=allow_reflection)
        score = _score_hypothesis(transform, cent_A, cent_B, M)
        hypotheses.append({"transform": transform, "score": score, "source": "global"})

    for a1, a2 in combinations(anchor_pool.tolist(), 2):
        target_cands_1 = sort_idx[a1, : min(3, sort_idx.shape[1])]
        target_cands_2 = sort_idx[a2, : min(3, sort_idx.shape[1])]
        for b1 in target_cands_1:
            for b2 in target_cands_2:
                if b1 == b2:
                    continue
                src_pts = cent_A[[a1, a2]]
                tgt_pts = cent_B[[b1, b2]]
                weights = np.ones(2, dtype=np.float64)
                for allow_reflection in (False, True):
                    transform = _fit_weighted_transform(src_pts, tgt_pts, weights, allow_reflection=allow_reflection)
                    score = _score_hypothesis(transform, cent_A, cent_B, M)
                    hypotheses.append(
                        {
                            "transform": transform,
                            "score": score,
                            "source": f"anchors:{a1},{a2}->{b1},{b2}",
                        }
                    )

    hypotheses.sort(key=lambda item: item["score"])
    unique: list[Dict[str, Any]] = []
    for item in hypotheses:
        if len(unique) >= max_hypotheses:
            break
        if _is_duplicate_transform(item["transform"], [u["transform"] for u in unique]):
            continue
        unique.append(item)
    return unique


def _score_hypothesis(transform: RigidTransform, cent_A: ArrayLike, cent_B: ArrayLike, desc_cost: ArrayLike) -> float:
    warped = transform.apply(cent_A)
    D = cdist(warped, cent_B)
    nearest = np.argmin(D, axis=1)
    geom = np.mean(np.min(D, axis=1))
    feature = np.mean(desc_cost[np.arange(desc_cost.shape[0]), nearest])

    if cent_A.shape[0] > 2:
        nn = NearestNeighbors(n_neighbors=min(4, cent_A.shape[0]))
        nn.fit(cent_A)
        _, idx = nn.kneighbors(cent_A)
        smooth_terms = []
        for i in range(cent_A.shape[0]):
            nbrs = idx[i, 1:]
            if nbrs.size == 0:
                continue
            smooth_terms.append(np.mean(np.linalg.norm(warped[i] - warped[nbrs], axis=1)))
        smooth = float(np.mean(smooth_terms)) if smooth_terms else 0.0
    else:
        smooth = 0.0

    return float(feature + geom / (np.median(D) + 1e-6) + 0.1 * smooth / (np.median(D) + 1e-6))


def _is_duplicate_transform(transform: RigidTransform, existing: Iterable[RigidTransform]) -> bool:
    for other in existing:
        mat_close = np.linalg.norm(transform.matrix - other.matrix) < 1e-3
        tr_close = np.linalg.norm(transform.translation - other.translation) < 1e-2
        if mat_close and tr_close:
            return True
    return False


def _fit_weighted_transform(
    src_pts: ArrayLike,
    tgt_pts: ArrayLike,
    weights: ArrayLike,
    allow_reflection: bool = False,
) -> RigidTransform:
    src_pts = np.asarray(src_pts, dtype=np.float64)
    tgt_pts = np.asarray(tgt_pts, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)

    src_cent = np.sum(weights[:, None] * src_pts, axis=0)
    tgt_cent = np.sum(weights[:, None] * tgt_pts, axis=0)
    src_c = src_pts - src_cent
    tgt_c = tgt_pts - tgt_cent
    H = (weights[:, None] * src_c).T @ tgt_c
    U, _, Vt = np.linalg.svd(H, full_matrices=False)
    R = Vt.T @ U.T
    if not allow_reflection and np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = tgt_cent - src_cent @ R.T
    return RigidTransform(matrix=R.astype(np.float64), translation=t.astype(np.float64), reflected=bool(np.linalg.det(R) < 0))


def _fit_transform_from_coupling(
    src_pts: ArrayLike,
    tgt_pts: ArrayLike,
    coupling: ArrayLike,
    allow_reflection: bool = False,
) -> RigidTransform:
    weights_src = coupling.sum(axis=1)
    mass = float(weights_src.sum())
    if mass <= 0:
        return RigidTransform(matrix=np.eye(2), translation=np.zeros(2), reflected=False)
    bary = (coupling @ tgt_pts) / (weights_src[:, None] + 1e-12)
    valid = weights_src > np.quantile(weights_src, 0.25)
    if valid.sum() < 3:
        valid = weights_src > 0
    return _fit_weighted_transform(src_pts[valid], bary[valid], weights_src[valid], allow_reflection=allow_reflection)


def _type_fallback_map(
    expr_src: ArrayLike,
    expr_tgt: ArrayLike,
    labels_src: ArrayLike,
    labels_tgt: ArrayLike,
) -> Dict[str, Optional[str]]:
    uniq_src = np.unique(labels_src)
    uniq_tgt = np.unique(labels_tgt)
    means_tgt = {label: expr_tgt[labels_tgt == label].mean(axis=0) for label in uniq_tgt}
    means_tgt_norm = {label: vec / (np.linalg.norm(vec) + 1e-12) for label, vec in means_tgt.items()}
    fallback: Dict[str, Optional[str]] = {}
    for label in uniq_src:
        if label in means_tgt_norm:
            fallback[label] = label
            continue
        src_mean = expr_src[labels_src == label].mean(axis=0)
        src_mean = src_mean / (np.linalg.norm(src_mean) + 1e-12)
        best_label = None
        best_score = np.inf
        for target_label, target_mean in means_tgt_norm.items():
            score = 1.0 - float(src_mean @ target_mean)
            if score < best_score:
                best_score = score
                best_label = target_label
        fallback[label] = best_label
    return fallback


def _refine_hypothesis(
    sliceA: AnnData,
    sliceB: AnnData,
    expr_A_n: ArrayLike,
    expr_B_n: ArrayLike,
    nbhd_A: ArrayLike,
    nbhd_B: ArrayLike,
    coords_A: ArrayLike,
    coords_B: ArrayLike,
    labels_A: ArrayLike,
    labels_B: ArrayLike,
    graph_A: sp.csr_matrix,
    graph_B: sp.csr_matrix,
    neighbor_idx_A: ArrayLike,
    neighbor_idx_B: ArrayLike,
    spatial_scale: float,
    boundary_A: ArrayLike,
    boundary_B: ArrayLike,
    type_fallback_A2B: Dict[str, Optional[str]],
    type_fallback_B2A: Dict[str, Optional[str]],
    mode: str,
    init_transform: RigidTransform,
    candidate_k: int,
    n_iters: int,
    verbose: bool = False,
) -> AlignmentResult:
    del graph_A, graph_B

    transform = RigidTransform(
        matrix=init_transform.matrix.copy(),
        translation=init_transform.translation.copy(),
        reflected=init_transform.reflected,
    )
    deformation: Optional[DeformationField] = None

    src_warped = transform.apply(coords_A)
    candidate_fwd = _build_candidate_graph(src_warped, coords_B, labels_A, labels_B, candidate_k, type_fallback_A2B)
    candidate_rev = _build_candidate_graph(coords_B, src_warped, labels_B, labels_A, candidate_k, type_fallback_B2A)

    fwd_pack = _update_sparse_transport(
        src_coords=src_warped,
        tgt_coords=coords_B,
        expr_src=expr_A_n,
        expr_tgt=expr_B_n,
        nbhd_src=nbhd_A,
        nbhd_tgt=nbhd_B,
        labels_src=labels_A,
        labels_tgt=labels_B,
        neighbor_idx=neighbor_idx_A,
        candidate_indices=candidate_fwd,
        prev_bary=None,
        reverse_bary=None,
        temperature=0.35,
        unmatched_confidence=boundary_A,
        spatial_scale=spatial_scale,
        lambda_dynamic=0.0,
        sharpen=False,
    )
    rev_pack = _update_sparse_transport(
        src_coords=coords_B,
        tgt_coords=src_warped,
        expr_src=expr_B_n,
        expr_tgt=expr_A_n,
        nbhd_src=nbhd_B,
        nbhd_tgt=nbhd_A,
        labels_src=labels_B,
        labels_tgt=labels_A,
        neighbor_idx=neighbor_idx_B,
        candidate_indices=candidate_rev,
        prev_bary=None,
        reverse_bary=None,
        temperature=0.35,
        unmatched_confidence=boundary_B,
        spatial_scale=spatial_scale,
        lambda_dynamic=0.0,
        sharpen=False,
    )

    for iter_idx in range(n_iters):
        frac = (iter_idx + 1) / max(n_iters, 1)
        temperature = 0.35 * (1.0 - frac) + 0.08 * frac
        lambda_dynamic = 0.6 * frac
        if verbose:
            print(f"[coherent] iter={iter_idx + 1} temp={temperature:.3f}")

        positive = fwd_pack["row_mass"] > 0
        if np.any(positive):
            transform = _fit_weighted_transform(
                coords_A[positive],
                fwd_pack["barycenters"][positive],
                np.maximum(fwd_pack["row_mass"][positive], 1e-12),
                allow_reflection=True,
            )
        src_warped = transform.apply(coords_A)

        if mode == "cross_timepoint" and iter_idx >= 1:
            deformation = _fit_deformation_field(src_warped, fwd_pack["barycenters"], fwd_pack["row_mass"])
            if deformation is not None:
                src_warped = deformation.apply(src_warped)

        candidate_fwd = _build_candidate_graph(src_warped, coords_B, labels_A, labels_B, candidate_k, type_fallback_A2B)
        candidate_rev = _build_candidate_graph(coords_B, src_warped, labels_B, labels_A, candidate_k, type_fallback_B2A)

        fwd_pack = _update_sparse_transport(
            src_coords=src_warped,
            tgt_coords=coords_B,
            expr_src=expr_A_n,
            expr_tgt=expr_B_n,
            nbhd_src=nbhd_A,
            nbhd_tgt=nbhd_B,
            labels_src=labels_A,
            labels_tgt=labels_B,
            neighbor_idx=neighbor_idx_A,
            candidate_indices=candidate_fwd,
            prev_bary=fwd_pack["barycenters"],
            reverse_bary=rev_pack["barycenters"],
            temperature=temperature,
            unmatched_confidence=boundary_A,
            spatial_scale=spatial_scale,
            lambda_dynamic=lambda_dynamic,
            sharpen=iter_idx == n_iters - 1,
        )
        rev_pack = _update_sparse_transport(
            src_coords=coords_B,
            tgt_coords=src_warped,
            expr_src=expr_B_n,
            expr_tgt=expr_A_n,
            nbhd_src=nbhd_B,
            nbhd_tgt=nbhd_A,
            labels_src=labels_B,
            labels_tgt=labels_A,
            neighbor_idx=neighbor_idx_B,
            candidate_indices=candidate_rev,
            prev_bary=rev_pack["barycenters"],
            reverse_bary=fwd_pack["barycenters"],
            temperature=temperature,
            unmatched_confidence=boundary_B,
            spatial_scale=spatial_scale,
            lambda_dynamic=lambda_dynamic,
            sharpen=iter_idx == n_iters - 1,
        )

    result = AlignmentResult(
        pi_fwd=fwd_pack["pi"].tocsr(),
        pi_rev=rev_pack["pi"].tocsr(),
        raw_unmatched_src=fwd_pack["unmatched_mass"],
        raw_unmatched_tgt=rev_pack["unmatched_mass"],
        selected_hypothesis={},
        hypothesis_scores=[],
        rigid_transform=transform,
        deformation_field=deformation,
        src_warped=src_warped,
        tgt_warped=np.asarray(coords_B, dtype=np.float64),
        forward_barycenters=fwd_pack["barycenters"],
        reverse_barycenters=rev_pack["barycenters"],
        metrics={},
    )
    result.metrics = summarize_alignment_metrics(sliceA, sliceB, result)
    return result


def _build_candidate_graph(
    src_coords: ArrayLike,
    tgt_coords: ArrayLike,
    src_labels: ArrayLike,
    tgt_labels: ArrayLike,
    candidate_k: int,
    fallback_map: Dict[str, Optional[str]],
) -> list[ArrayLike]:
    candidate_graph: list[ArrayLike] = [np.empty(0, dtype=np.int64) for _ in range(src_coords.shape[0])]
    type_to_indices: Dict[str, ArrayLike] = {}
    type_to_nn: Dict[str, NearestNeighbors] = {}

    for label in np.unique(tgt_labels):
        idx = np.where(tgt_labels == label)[0]
        if idx.size == 0:
            continue
        type_to_indices[label] = idx
        n_neighbors = int(max(1, min(candidate_k, idx.size)))
        nn = NearestNeighbors(n_neighbors=n_neighbors)
        nn.fit(tgt_coords[idx])
        type_to_nn[label] = nn

    for label in np.unique(src_labels):
        src_idx = np.where(src_labels == label)[0]
        target_label = label if label in type_to_indices else fallback_map.get(label)
        if target_label is None or target_label not in type_to_indices:
            continue
        target_idx = type_to_indices[target_label]
        nn = type_to_nn[target_label]
        _, local = nn.kneighbors(src_coords[src_idx], n_neighbors=nn.n_neighbors)
        mapped = target_idx[local]
        for local_pos, global_pos in enumerate(src_idx):
            candidate_graph[global_pos] = mapped[local_pos].astype(np.int64)
    return candidate_graph


def _update_sparse_transport(
    src_coords: ArrayLike,
    tgt_coords: ArrayLike,
    expr_src: ArrayLike,
    expr_tgt: ArrayLike,
    nbhd_src: ArrayLike,
    nbhd_tgt: ArrayLike,
    labels_src: ArrayLike,
    labels_tgt: ArrayLike,
    neighbor_idx: ArrayLike,
    candidate_indices: list[ArrayLike],
    prev_bary: Optional[ArrayLike],
    reverse_bary: Optional[ArrayLike],
    temperature: float,
    unmatched_confidence: ArrayLike,
    spatial_scale: float,
    lambda_dynamic: float,
    sharpen: bool,
) -> Dict[str, Any]:
    n_src = src_coords.shape[0]
    n_tgt = tgt_coords.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    bary = np.zeros_like(src_coords)
    row_mass = np.zeros(n_src, dtype=np.float64)
    unmatched_mass = np.zeros(n_src, dtype=np.float64)

    neighbor_lists = neighbor_idx[:, : min(8, neighbor_idx.shape[1])] if neighbor_idx.ndim == 2 else neighbor_idx
    src_mass_unit = 1.0 / max(n_src, 1)

    for i in range(n_src):
        cands = candidate_indices[i]
        if cands.size == 0:
            bary[i] = src_coords[i]
            unmatched_mass[i] = src_mass_unit
            continue

        gene = _candidate_gene_cost(expr_src[i], expr_tgt[cands])
        niche = _jsd_to_candidates(nbhd_src[i], nbhd_tgt[cands])
        spatial = np.linalg.norm(tgt_coords[cands] - src_coords[i], axis=1) / (spatial_scale + 1e-12)
        spatial = np.clip(spatial, 0.0, 4.0) / 4.0
        type_penalty = 4.0 * (labels_tgt[cands] != labels_src[i]).astype(np.float64)
        feature_cost = gene + niche + spatial + type_penalty

        compact_cost = np.zeros_like(feature_cost)
        smooth_cost = np.zeros_like(feature_cost)
        isometry_cost = np.zeros_like(feature_cost)
        cycle_cost = np.zeros_like(feature_cost)

        if prev_bary is not None:
            compact_cost = np.linalg.norm(tgt_coords[cands] - prev_bary[i], axis=1) / (spatial_scale + 1e-12)
            compact_cost = np.clip(compact_cost, 0.0, 4.0) / 4.0

            nbrs = neighbor_lists[i]
            valid_nbrs = nbrs[(nbrs >= 0) & (nbrs < prev_bary.shape[0])]
            if valid_nbrs.size > 0:
                nbr_bary = prev_bary[valid_nbrs]
                nbr_src = src_coords[valid_nbrs]
                smooth = cdist(tgt_coords[cands], nbr_bary)
                smooth_cost = np.mean(np.clip(smooth / (spatial_scale + 1e-12), 0.0, 4.0) / 4.0, axis=1)

                src_edge = np.linalg.norm(nbr_src - src_coords[i], axis=1) / (spatial_scale + 1e-12)
                tgt_edge = cdist(tgt_coords[cands], nbr_bary) / (spatial_scale + 1e-12)
                isometry_cost = np.mean(np.abs(tgt_edge - src_edge[None, :]), axis=1)

        if reverse_bary is not None:
            cycle = np.linalg.norm(reverse_bary[cands] - src_coords[i], axis=1) / (spatial_scale + 1e-12)
            cycle_cost = np.clip(cycle, 0.0, 4.0) / 4.0

        total_cost = feature_cost + lambda_dynamic * (compact_cost + smooth_cost + isometry_cost + cycle_cost)
        unmatched_cost = 0.35 + 0.85 * unmatched_confidence[i]
        probs = _softmax_neg(np.concatenate([total_cost, [unmatched_cost]]), temperature)

        unmatched_prob = float(probs[-1])
        match_probs = probs[:-1]
        if sharpen and match_probs.size > 0:
            keep = np.argsort(-match_probs)[: min(2, match_probs.size)]
            sharpened = np.zeros_like(match_probs)
            sharpened[keep] = match_probs[keep] ** 2
            denom = sharpened.sum()
            if denom > 0:
                match_probs = sharpened / denom

        matched_mass = src_mass_unit * (1.0 - unmatched_prob)
        row_mass[i] = matched_mass
        unmatched_mass[i] = src_mass_unit - matched_mass

        if matched_mass <= 0:
            bary[i] = src_coords[i]
            continue

        weights = match_probs / (match_probs.sum() + 1e-12)
        bary[i] = weights @ tgt_coords[cands]
        active = np.where(weights > 1e-4)[0]
        if active.size == 0:
            active = np.array([int(np.argmax(weights))])
        for a in active:
            rows.append(i)
            cols.append(int(cands[a]))
            data.append(float(matched_mass * weights[a]))

    pi = sp.csr_matrix((np.asarray(data), (np.asarray(rows), np.asarray(cols))), shape=(n_src, n_tgt))
    return {
        "pi": pi,
        "barycenters": bary,
        "row_mass": row_mass,
        "unmatched_mass": unmatched_mass,
    }


def _fit_deformation_field(src_warped: ArrayLike, bary: ArrayLike, weights: ArrayLike) -> Optional[DeformationField]:
    if not np.any(weights > 0):
        return None
    positive = weights[weights > 0]
    valid = weights > np.quantile(positive, 0.35)
    if valid.sum() < 12:
        return None

    src_sel = src_warped[valid]
    tgt_sel = bary[valid]
    max_controls = min(128, src_sel.shape[0])
    step = max(1, src_sel.shape[0] // max_controls)
    ctrl = src_sel[::step][:max_controls]
    target = tgt_sel[::step][:max_controls]
    if ctrl.shape[0] < 8:
        return None

    sqdist = cdist(ctrl, ctrl, metric="sqeuclidean")
    med = float(np.median(np.sqrt(sqdist[sqdist > 0]))) if np.any(sqdist > 0) else 1.0
    gamma = 1.0 / (2.0 * (med ** 2 + 1e-12))
    K = np.exp(-gamma * sqdist)
    residual = target - ctrl
    lam = 1e-2
    try:
        W = np.linalg.solve(K + lam * np.eye(K.shape[0]), residual)
    except np.linalg.LinAlgError:
        W = np.linalg.lstsq(K + lam * np.eye(K.shape[0]), residual, rcond=None)[0]
    return DeformationField(control_points=ctrl, weights=W.astype(np.float64), gamma=float(gamma))


def _candidate_gene_cost(src_row: ArrayLike, tgt_rows: ArrayLike) -> ArrayLike:
    dots = np.clip(tgt_rows @ src_row, -1.0, 1.0)
    return 0.5 * (1.0 - dots)


def _jsd_to_candidates(src_row: ArrayLike, tgt_rows: ArrayLike) -> ArrayLike:
    p = np.asarray(src_row, dtype=np.float64)
    q = np.asarray(tgt_rows, dtype=np.float64)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum(axis=1, keepdims=True) + 1e-12)
    m = 0.5 * (q + p[None, :])
    kl_pm = np.sum(p[None, :] * np.log((p[None, :] + 1e-12) / (m + 1e-12)), axis=1)
    kl_qm = np.sum(q * np.log((q + 1e-12) / (m + 1e-12)), axis=1)
    return np.sqrt(0.5 * (kl_pm + kl_qm))


def _softmax_neg(costs: ArrayLike, temperature: float) -> ArrayLike:
    scaled = -np.asarray(costs, dtype=np.float64) / max(temperature, 1e-3)
    scaled = scaled - np.max(scaled)
    expv = np.exp(scaled)
    return expv / (expv.sum() + 1e-12)


def _sparse_type_match(pi: sp.csr_matrix, src_types: ArrayLike, tgt_types: ArrayLike) -> float:
    total = float(pi.sum())
    if total <= 0:
        return 0.0
    matches = 0.0
    for i in range(pi.shape[0]):
        start, end = pi.indptr[i], pi.indptr[i + 1]
        cols = pi.indices[start:end]
        vals = pi.data[start:end]
        if cols.size == 0:
            continue
        matches += float(vals[tgt_types[cols] == src_types[i]].sum())
    return 100.0 * matches / total


def _sparse_entropy_support(pi: sp.csr_matrix) -> tuple[float, float]:
    total = float(pi.sum())
    if total <= 0:
        return 0.0, 0.0
    vals = pi.data / total
    entropy = float(-np.sum(vals * np.log(vals + 1e-12)))
    max_entropy = float(np.log(max(pi.nnz, 1)))
    entropy_pct = 100.0 * entropy / (max_entropy + 1e-12) if max_entropy > 0 else 0.0

    supports = []
    for i in range(pi.shape[0]):
        start, end = pi.indptr[i], pi.indptr[i + 1]
        row_vals = pi.data[start:end]
        row_sum = float(row_vals.sum())
        if row_sum <= 0:
            continue
        cond = row_vals / row_sum
        supports.append(float(np.exp(-np.sum(cond * np.log(cond + 1e-12)))))
    eff = float(np.mean(supports)) if supports else 0.0
    return entropy_pct, eff


def _sparse_symmetry_ambiguity(pi: sp.csr_matrix) -> float:
    ratios = []
    for i in range(pi.shape[0]):
        start, end = pi.indptr[i], pi.indptr[i + 1]
        row_vals = np.sort(pi.data[start:end])
        if row_vals.size == 0:
            continue
        top1 = row_vals[-1]
        top2 = row_vals[-2] if row_vals.size > 1 else 0.0
        ratios.append(float(top2 / (top1 + 1e-12)))
    return float(np.mean(ratios)) if ratios else 0.0


def _sparse_best_match(pi: sp.csr_matrix) -> ArrayLike:
    best = np.full(pi.shape[0], -1, dtype=np.int64)
    for i in range(pi.shape[0]):
        start, end = pi.indptr[i], pi.indptr[i + 1]
        row_vals = pi.data[start:end]
        if row_vals.size == 0:
            continue
        row_cols = pi.indices[start:end]
        best[i] = int(row_cols[int(np.argmax(row_vals))])
    return best


def _safe_weighted_rmse(src: ArrayLike, tgt: ArrayLike, weights: ArrayLike) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    if not np.any(weights > 0):
        return float("nan")
    residual = np.linalg.norm(src - tgt, axis=1)
    return float(np.sqrt(np.average(residual ** 2, weights=weights + 1e-12)))


def _nsp_score(src_warped: ArrayLike, tgt_coords: ArrayLike, best_targets: ArrayLike, valid_best: ArrayLike, k_nn: int) -> float:
    if src_warped.shape[0] < 2 or tgt_coords.shape[0] < 2:
        return float("nan")
    nn_src = NearestNeighbors(n_neighbors=min(k_nn, src_warped.shape[0])).fit(src_warped)
    nn_tgt = NearestNeighbors(n_neighbors=min(k_nn, tgt_coords.shape[0])).fit(tgt_coords)
    _, idx_src = nn_src.kneighbors(src_warped)
    _, idx_tgt = nn_tgt.kneighbors(tgt_coords)

    scores = []
    for i in range(src_warped.shape[0]):
        if not valid_best[i]:
            continue
        j = best_targets[i]
        mapped_nbrs = {best_targets[n] for n in idx_src[i] if valid_best[n]}
        true_nbrs = set(idx_tgt[j])
        if not mapped_nbrs or not true_nbrs:
            continue
        scores.append(len(mapped_nbrs & true_nbrs) / (len(mapped_nbrs | true_nbrs) + 1e-12))
    return 100.0 * float(np.mean(scores)) if scores else float("nan")


def _mapped_geometry_consistency(
    src_warped: ArrayLike,
    bary: ArrayLike,
    row_mass: ArrayLike,
    max_points: int = 512,
) -> tuple[float, float]:
    if not np.any(row_mass > 0):
        return float("nan"), float("nan")
    positive = row_mass[row_mass > 0]
    valid = row_mass > np.quantile(positive, 0.35)
    idx = np.where(valid)[0]
    if idx.size < 6:
        return float("nan"), float("nan")
    if idx.size > max_points:
        idx = idx[np.linspace(0, idx.size - 1, max_points, dtype=int)]

    src_sel = src_warped[idx]
    bary_sel = bary[idx]
    D_src = cdist(src_sel, src_sel)
    D_bary = cdist(bary_sel, bary_sel)
    tri = np.triu_indices(src_sel.shape[0], k=1)
    x = D_src[tri]
    y = D_bary[tri]
    if x.size < 6:
        return float("nan"), float("nan")
    x = x / (np.median(x[x > 0]) + 1e-12)
    y = y / (np.median(y[y > 0]) + 1e-12)
    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
    stress = float(np.sqrt(np.sum((x - y) ** 2) / (np.sum(x ** 2) + 1e-12)))
    return corr, stress


def _compactness_score(pi: sp.csr_matrix, target_coords: ArrayLike, bary: ArrayLike) -> float:
    numer = 0.0
    denom = 0.0
    for i in range(pi.shape[0]):
        start, end = pi.indptr[i], pi.indptr[i + 1]
        cols = pi.indices[start:end]
        vals = pi.data[start:end]
        if cols.size == 0:
            continue
        diff = np.linalg.norm(target_coords[cols] - bary[i], axis=1)
        numer += float(np.sum(vals * diff))
        denom += float(np.sum(vals))
    return numer / (denom + 1e-12)


def _cycle_error(pi_fwd: sp.csr_matrix, reverse_bary: ArrayLike, src_warped: ArrayLike) -> float:
    errors = []
    for i in range(pi_fwd.shape[0]):
        start, end = pi_fwd.indptr[i], pi_fwd.indptr[i + 1]
        cols = pi_fwd.indices[start:end]
        vals = pi_fwd.data[start:end]
        if cols.size == 0:
            continue
        weights = vals / (vals.sum() + 1e-12)
        cycle_pos = weights @ reverse_bary[cols]
        errors.append(np.linalg.norm(cycle_pos - src_warped[i]))
    return float(np.mean(errors)) if errors else float("nan")


def _selection_score(metrics: Dict[str, float]) -> float:
    def safe(value: float, fallback: float) -> float:
        return float(value) if np.isfinite(value) else fallback

    penalty = 0.0
    penalty += 0.4 * (100.0 - metrics["cell_type_match_pct"]) / 100.0
    penalty += 0.8 * safe(metrics["spatial_rmse"], 2.0)
    penalty += 0.8 * (100.0 - safe(metrics["nsp_pct"], 0.0)) / 100.0
    penalty += 0.6 * metrics["symmetry_ambiguity"]
    penalty += 0.6 * safe(metrics["forward_compactness"], 1.0)
    penalty += 0.6 * safe(metrics["reverse_compactness"], 1.0)
    penalty += 0.6 * safe(metrics["cycle_error"], 1.0)
    penalty += 0.8 * safe(metrics["mapped_geom_stress"], 1.0)
    penalty += 0.2 * metrics["eff_targets_per_source"]
    penalty += 0.2 * metrics["eff_sources_per_target"]
    penalty += 0.5 * metrics["unmatched_src_mass_pct"] / 100.0
    return float(penalty)
