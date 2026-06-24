from __future__ import annotations

import numpy as np
import fpsample
import pysuperansac


def _fps_sample(points, n_samples, h=5, min_h=3):
    n_samples = int(n_samples)
    if n_samples > 0:
        h = min(h, int(np.log2(n_samples)))
    h = max(min_h, h)
    return fpsample.bucket_fps_kdline_sampling(points, n_samples, h=h)


def are_scales_valid(s1, s2, s3):
    if s1 <= 0 or s2 <= 0 or s3 <= 0:
        return False
    max_scale = max(s1, s2, s3)
    min_scale = min(s1, s2, s3)
    aspect_ratio = max_scale / (min_scale + 1e-10)
    return max_scale <= 10.0 and min_scale >= 0.05 and aspect_ratio < 2.5


def voxel_clean_indices(points, grid_size=16, min_count_per_voxel=2):
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    voxel_size = (max_bound - min_bound) / grid_size
    voxel_size[voxel_size == 0] = 1e-6

    voxel_indices = np.floor((points - min_bound) / voxel_size).astype(int)
    voxel_indices = np.clip(voxel_indices, 0, grid_size - 1)
    linear_idx = (
        voxel_indices[:, 0] * grid_size * grid_size
        + voxel_indices[:, 1] * grid_size
        + voxel_indices[:, 2]
    )
    _, inverse, counts = np.unique(linear_idx, return_inverse=True, return_counts=True)
    valid_voxels = counts >= min_count_per_voxel
    return valid_voxels[inverse]


def geometric_consensus_score(src_pts, tgt_pts, beta, n_iter=50, hist_bins=60):
    M = len(src_pts)
    if M < 3 or beta <= 0:
        return np.ones(M)
    src = np.asarray(src_pts, dtype=float)
    tgt = np.asarray(tgt_pts, dtype=float)
    Dsrc = np.linalg.norm(src[:, None, :] - src[None, :, :], axis=2)
    Dtgt = np.linalg.norm(tgt[:, None, :] - tgt[None, :, :], axis=2)
    eps = 1e-6
    logr = np.log(np.maximum(Dtgt, eps)) - np.log(np.maximum(Dsrc, eps))
    iu = np.triu_indices(M, 1)
    vals = logr[iu]
    finite = vals[np.isfinite(vals)]
    if finite.size == 0 or np.ptp(finite) < 1e-9:
        return np.ones(M)
    hist, edges = np.histogram(finite, bins=hist_bins)
    kbin = int(np.argmax(hist))
    s_hat = 0.5 * (edges[kbin] + edges[kbin + 1])
    C = (np.abs(logr - s_hat) < beta).astype(float)
    np.fill_diagonal(C, 0.0)
    e = np.ones(M) / np.sqrt(M)
    for _ in range(n_iter):
        e = C @ e
        n = np.linalg.norm(e)
        if n < 1e-12:
            break
        e = e / n
    return np.abs(e)


def _axis_aligned_consensus_score(
    src_pts,
    tgt_pts,
    beta,
    aniso_shrink=1.0,
    n_iter=50,
    hist_bins=60,
):
    M = len(src_pts)
    if M < 4 or beta <= 0:
        return np.ones(M)
    src = np.asarray(src_pts, dtype=float)
    tgt = np.asarray(tgt_pts, dtype=float)
    eps = 1e-9

    iu = np.triu_indices(M, 1)
    d = src[iu[0]] - src[iu[1]]
    y = np.sum((tgt[iu[0]] - tgt[iu[1]]) ** 2, axis=1)
    ld = np.sum(d * d, axis=1)
    valid = ld > 1e-12
    if int(valid.sum()) < 6:
        return geometric_consensus_score(src, tgt, beta, n_iter=n_iter, hist_bins=hist_bins)
    d, y, ld = d[valid], y[valid], ld[valid]

    lr = 0.5 * np.log(np.maximum(y, eps)) - 0.5 * np.log(np.maximum(ld, eps))
    if np.ptp(lr) < 1e-9:
        log_shat = float(np.median(lr))
    else:
        hist, edges = np.histogram(lr, bins=hist_bins)
        kbin = int(np.argmax(hist))
        log_shat = 0.5 * (edges[kbin] + edges[kbin + 1])

    g2 = y / ld
    feat = (d * d) / ld[:, None]
    s = np.full(3, float(np.median(g2)))
    w = np.ones(len(g2))
    for _ in range(5):
        WF = feat * w[:, None]
        try:
            s = np.linalg.solve(feat.T @ WF + 1e-9 * np.eye(3), WF.T @ g2)
        except np.linalg.LinAlgError:
            s = np.linalg.lstsq(feat.T @ WF, WF.T @ g2, rcond=None)[0]
        s = np.maximum(s, eps)
        r = np.log(np.maximum(g2, eps)) - np.log(np.maximum(feat @ s, eps))
        med = np.median(r)
        mad = np.median(np.abs(r - med)) + eps
        c = 1.345 * 1.4826 * mad
        w = np.where(np.abs(r - med) <= c, 1.0, c / np.maximum(np.abs(r - med), eps))

    if aniso_shrink < 1.0:
        ls = 0.5 * np.log(np.maximum(s, eps))
        ls = log_shat + aniso_shrink * (ls - log_shat)
        s = np.exp(2.0 * ls)

    Dvec = src[:, None, :] - src[None, :, :]
    pred2 = np.maximum((Dvec ** 2) @ s, eps)
    Dtgt2 = np.sum((tgt[:, None, :] - tgt[None, :, :]) ** 2, axis=2)
    resid = (Dtgt2 - pred2) / pred2
    C = (np.abs(resid) < beta).astype(float)
    np.fill_diagonal(C, 0.0)
    e = np.ones(M) / np.sqrt(M)
    for _ in range(n_iter):
        e = C @ e
        n = np.linalg.norm(e)
        if n < 1e-12:
            break
        e = e / n
    return np.abs(e)


def geometric_consensus_mask(src_pts, tgt_pts, beta, rel_thresh=0.05, aniso_shrink=1.0):
    M = len(src_pts)
    if M < 3 or beta <= 0:
        return np.ones(M, dtype=bool)
    e = _axis_aligned_consensus_score(
        src_pts,
        tgt_pts,
        beta,
        aniso_shrink=aniso_shrink,
    )
    emax = e.max()
    if emax <= 0:
        return np.zeros(M, dtype=bool)
    return e >= rel_thresh * emax


def _cap_correspondences(corres, weights, src_points, max_correspondences):
    M = weights.size
    if max_correspondences is None or max_correspondences <= 0 or M <= max_correspondences:
        return corres, weights

    if src_points is not None:
        pts = np.ascontiguousarray(src_points[corres[:, 0]], dtype=np.float64)
        keep = _fps_sample(pts, max_correspondences, h=5)
    else:
        keep = np.argpartition(-weights, max_correspondences - 1)[:max_correspondences]
    return corres[keep], weights[keep]


def find_correspondences_mknn(
    src_features,
    tgt_features,
    src_points=None,
    tgt_points=None,
    n_points=4096,
    max_correspondences=512,
    normalize_weights=True,
    k=1,
):
    Ns, Nt = src_features.shape[0], tgt_features.shape[0]
    if Ns == 0 or Nt == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float32)

    src_fps = (
        _fps_sample(src_points, n_points, h=5)
        if (src_points is not None and Ns > n_points)
        else np.arange(Ns)
    )
    tgt_fps = (
        _fps_sample(tgt_points, n_points, h=5)
        if (tgt_points is not None and Nt > n_points)
        else np.arange(Nt)
    )

    sf = src_features[src_fps]
    tf = tgt_features[tgt_fps]
    ns, nt = sf.shape[0], tf.shape[0]
    sim = sf @ tf.T

    k_t = min(max(1, int(k)), nt)
    k_s = min(max(1, int(k)), ns)
    topk_t = np.argpartition(-sim, k_t - 1, axis=1)[:, :k_t]
    topk_s = np.argpartition(-sim, k_s - 1, axis=0)[:k_s, :]

    src_ids = np.repeat(np.arange(ns), k_t)
    tgt_ids = topk_t.reshape(-1)
    cand_w = sim[src_ids, tgt_ids]
    mutual = (topk_s[:, tgt_ids] == src_ids[None, :]).any(axis=0)
    src_ids = src_ids[mutual]
    tgt_ids = tgt_ids[mutual]
    cand_w = cand_w[mutual]
    if src_ids.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros(0, dtype=np.float32)

    order = np.argsort(-cand_w)
    used_src = np.zeros(ns, dtype=bool)
    used_tgt = np.zeros(nt, dtype=bool)
    keep = []
    for idx in order:
        r = src_ids[idx]
        c = tgt_ids[idx]
        if used_src[r] or used_tgt[c]:
            continue
        used_src[r] = True
        used_tgt[c] = True
        keep.append(idx)
    keep = np.asarray(keep, dtype=np.int64)

    weights = cand_w[keep]
    corres = np.stack(
        [src_fps[src_ids[keep]], tgt_fps[tgt_ids[keep]]],
        axis=1,
    ).astype(np.int64)

    corres, weights = _cap_correspondences(
        corres,
        weights,
        src_points,
        max_correspondences,
    )
    if normalize_weights and weights.size > 0:
        weights = weights / (weights.sum() + 1e-10)
    return corres, weights


def run_ransac_registration(
    source_points,
    target_points,
    corres,
    weights,
    inlier_threshold=0.11,
    max_iterations=100000,
):
    config = pysuperansac.RANSACSettings()
    config.inlier_threshold = inlier_threshold
    config.min_iterations = 500
    config.max_iterations = max_iterations
    config.confidence = 0.999
    config.sampler = pysuperansac.SamplerType.PROSAC
    config.scoring = pysuperansac.ScoringType.MAGSAC
    config.local_optimization = pysuperansac.LocalOptimizationType.Nothing
    config.final_optimization = pysuperansac.LocalOptimizationType.Nothing

    filled_corres = np.concatenate(
        [source_points[corres[:, 0]], target_points[corres[:, 1]]],
        axis=1,
    )
    min_coordinates = np.min(filled_corres, axis=0)
    T1 = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
        [-min_coordinates[0], -min_coordinates[1], -min_coordinates[2], 1],
    ])
    T2inv = np.array([
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
        [min_coordinates[3], min_coordinates[4], min_coordinates[5], 1],
    ])
    transformed_correspondences = filled_corres - min_coordinates

    try:
        T_sr, inliers, score, _ = pysuperansac.estimateRigidTransform(
            np.ascontiguousarray(transformed_correspondences),
            np.ascontiguousarray(np.max(transformed_correspondences, axis=0)),
            np.ascontiguousarray(weights),
            config,
        )
    except TypeError as exc:
        origin = getattr(pysuperansac, "__file__", "unknown")
        raise RuntimeError(
            "Incompatible pysuperansac binding loaded from "
            f"{origin}. SUFLECA requires its vendored four-argument rigid-transform API. "
            "Remove stale pysuperansac .pth/path overrides and reinstall "
            "third_party/superansac into the active environment."
        ) from exc

    mask = np.zeros((transformed_correspondences.shape[0], 1), dtype=np.uint8)
    mask[inliers] = 1
    if T_sr is None:
        T_sr = np.eye(4)
    else:
        T_sr = T1 @ T_sr @ T2inv
        T_sr = T_sr.T
    return T_sr, mask, score


def solve_anisotropic_procrustes(src_points, tgt_points, corres, unbiased_info=False):
    src_m = src_points[corres[:, 0]]
    tgt_m = tgt_points[corres[:, 1]]
    src_mean = src_m.mean(axis=0)
    tgt_mean = tgt_m.mean(axis=0)
    src_c = src_m - src_mean
    tgt_c = tgt_m - tgt_mean

    src_cov = src_c.T @ src_c
    cross_cov = tgt_c.T @ src_c
    A = cross_cov @ np.linalg.pinv(src_cov)
    U, _, Vt = np.linalg.svd(A)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt

    Rt_tgt_c = (R.T @ tgt_c.T).T
    s_diag = np.zeros(3)
    for k in range(3):
        num = Rt_tgt_c[:, k] @ src_c[:, k]
        den = src_c[:, k] @ src_c[:, k]
        s_diag[k] = num / (den + 1e-10)
    S = np.diag(s_diag)
    t = tgt_mean - R @ S @ src_mean

    T = np.eye(4)
    T[:3, :3] = R @ S
    T[:3, 3] = t
    info = compute_information_matrix(R, S, t, src_m, tgt_m, unbiased=unbiased_info)
    return T, R, S, t, info


def compute_information_matrix(R, S, t, src_pts, tgt_pts, unbiased=False):
    M = src_pts.shape[0]
    t_flat = np.asarray(t).flatten()
    RS = R @ S
    q = (RS @ src_pts.T).T
    residuals = tgt_pts - (q + t_flat)

    J = np.zeros((M, 3, 9))
    J[:, 0, 1] = -q[:, 2]
    J[:, 0, 2] = q[:, 1]
    J[:, 1, 0] = q[:, 2]
    J[:, 1, 2] = -q[:, 0]
    J[:, 2, 0] = -q[:, 1]
    J[:, 2, 1] = q[:, 0]
    J[:, 0, 3] = -1.0
    J[:, 1, 4] = -1.0
    J[:, 2, 5] = -1.0
    J[:, :, 6:9] = -R[np.newaxis, :, :] * src_pts[:, np.newaxis, :]
    info = np.einsum("ijk,ijl->kl", J, J)

    if unbiased:
        dof = max(1.0, 3.0 * M - 9.0)
        sigma_sq = max(1e-15, float(np.sum(residuals ** 2) / dof))
        info = info / sigma_sq
    return info


def decompose_transform(T):
    A = T[:3, :3]
    t = T[:3, 3]
    U, _, Vt = np.linalg.svd(A)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    s_diag = np.array([R[:, k] @ A[:, k] for k in range(3)])
    return R, np.diag(s_diag), t
