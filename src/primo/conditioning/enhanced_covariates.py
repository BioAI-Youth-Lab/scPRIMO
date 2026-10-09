from __future__ import annotations

import numpy as np

from primo.llm.similar_utils import (
    _extract_similars_drug_name,
    _transform_similars_score_to_weight,
    extract_similars_candidates,
)


def _to_dense_float32(x):
    if hasattr(x, "toarray"):
        return x.toarray().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _extract_rows_dense_float32(x, row_mask):
    x_sub = x[row_mask]
    if hasattr(x_sub, "toarray"):
        return x_sub.toarray().astype(np.float32)
    return np.asarray(x_sub, dtype=np.float32)


def _resolve_control_mask(adata_train, control_col=None, control_label=None):
    if control_col is None:
        raise ValueError(
            "control_col is required when using control-derived statistics. "
            "For sciplex, this is often 'product_name_clean'."
        )
    if control_col not in adata_train.obs.columns:
        raise ValueError(
            f"control_col '{control_col}' not found in adata_train.obs. "
            f"Available columns include: {list(adata_train.obs.columns[:10])}"
        )

    control_series = adata_train.obs[control_col].astype(str)
    if control_label is None:
        candidates = ("vehicle", "control", "ctrl", "dmso", "untreated")
        lower = control_series.str.lower()
        matched = [
            c for c in candidates if (lower == c).any() or lower.str.contains(c).any()
        ]
        if len(matched) == 0:
            raise ValueError(
                "Unable to infer control_label automatically. "
                "Please pass control_label explicitly."
            )
        control_label = matched[0]

    control_label_str = str(control_label)
    lower = control_series.str.lower()
    label_lower = control_label_str.lower()
    exact_mask = (lower == label_lower).values
    contains_mask = lower.str.contains(label_lower, regex=False).values
    control_mask = exact_mask if exact_mask.any() else contains_mask

    if not control_mask.any():
        raise ValueError(
            f"No control rows found for control_col='{control_col}' "
            f"and control_label='{control_label_str}'. "
            "If your training data was filtered (for example by good_smile_chemvae), "
            "pass control_adata=<unfiltered or control-only AnnData> to fit_train_stats_pca."
        )

    return control_mask, control_col, control_label_str


def _compute_control_stats_by_cell(
    X_ctrl,
    ctrl_cell_types,
    cell_type_categories,
    eps=1e-6,
):
    import numpy as _np

    global_mu = X_ctrl.mean(axis=0, dtype=_np.float32)
    global_sd = _np.clip(X_ctrl.std(axis=0, dtype=_np.float32), eps, None)

    control_stats_by_cell = {}
    control_count_by_cell = {}
    for ct in cell_type_categories:
        rows_ct = ctrl_cell_types == ct
        control_count_by_cell[ct] = int(rows_ct.sum())
        if rows_ct.any():
            mu_ct = X_ctrl[rows_ct].mean(axis=0, dtype=_np.float32)
            sd_ct = _np.clip(X_ctrl[rows_ct].std(axis=0, dtype=_np.float32), eps, None)
            control_stats_by_cell[ct] = (mu_ct, sd_ct)
        else:
            control_stats_by_cell[ct] = (global_mu.copy(), global_sd.copy())

    return control_stats_by_cell, control_count_by_cell


def _fit_cell_type_control_context(
    adata_train,
    *,
    cell_types,
    cell_type_categories,
    cell_type_col="cell_type",
    gene_layer="counts",
    control_col=None,
    control_label=None,
    control_adata=None,
    cell_type_pca_n=16,
    random_state=42,
):
    import numpy as _np
    from sklearn.decomposition import PCA as _PCA

    control_source = control_adata if control_adata is not None else adata_train
    control_mask, control_col, control_label_str = _resolve_control_mask(
        control_source,
        control_col=control_col,
        control_label=control_label,
    )
    source_cell_types = _np.asarray(
        control_source.obs[cell_type_col].astype(str).values
    )
    X_ctrl = _extract_rows_dense_float32(control_source.layers[gene_layer], control_mask)
    ctrl_cell_types = source_cell_types[control_mask]

    max_components = int(min(cell_type_pca_n, X_ctrl.shape[0], X_ctrl.shape[1]))
    if max_components < 1:
        raise ValueError(
            "cell_type_pca_n resolved to < 1 after clipping by available control rows."
        )

    pca = _PCA(
        n_components=max_components,
        svd_solver="randomized",
        random_state=random_state,
    ).fit(X_ctrl)
    ctrl_scores = pca.transform(X_ctrl).astype(_np.float32)

    global_proto = ctrl_scores.mean(axis=0, dtype=_np.float32)
    prototypes_by_cell = {}
    control_count_by_cell = {}
    for ct in cell_type_categories:
        rows_ct = ctrl_cell_types == ct
        control_count_by_cell[ct] = int(rows_ct.sum())
        if rows_ct.any():
            prototypes_by_cell[ct] = ctrl_scores[rows_ct].mean(
                axis=0, dtype=_np.float32
            )
        else:
            prototypes_by_cell[ct] = global_proto.copy()

    return {
        "mode": "control_pca",
        "dim": int(max_components),
        "categories": list(cell_type_categories),
        "control_col": control_col,
        "control_label": control_label_str,
        "control_pca": pca,
        "control_prototypes_by_cell": prototypes_by_cell,
        "control_count_by_cell": control_count_by_cell,
        "global_control_proto": global_proto,
    }


def _build_cell_context_matrix(cell_types, train_fit):
    import numpy as _np
    import pandas as _pd

    context_fit = train_fit.get("cell_context")
    if context_fit is None:
        categories = train_fit["cell_type_categories"]
        ct_series = _pd.Categorical(cell_types, categories=categories)
        return _pd.get_dummies(ct_series).values.astype(_np.float32)

    mode = context_fit.get("mode", "onehot")
    if mode == "onehot":
        categories = context_fit.get("categories") or train_fit["cell_type_categories"]
        ct_series = _pd.Categorical(cell_types, categories=categories)
        return _pd.get_dummies(ct_series).values.astype(_np.float32)

    if mode == "control_pca":
        prototypes = context_fit["control_prototypes_by_cell"]
        fallback = context_fit["global_control_proto"]
        rows = [prototypes.get(str(ct), fallback) for ct in _np.asarray(cell_types).astype(str)]
        return _np.stack(rows).astype(_np.float32)

    raise ValueError(f"Unsupported cell context mode: {mode}")


def fit_train_stats_pca(
    adata_train,
    cond_stage_data_train,
    n_components=64,
    cell_type_col="cell_type",
    dose_col="dose",
    gene_layer="counts",
    random_state=42,
    cell_type_rep="onehot",
    cell_type_pca_n=16,
    control_col=None,
    control_label=None,
    control_adata=None,
    residualize_to_control=False,
    residual_shrinkage=10.0,
):
    """
    Fit PCA on train-only gene-expression statistics grouped by (cell_type, perturb_name).

    Returns a dict containing:
      - pca: fitted sklearn PCA on concatenated [group_mean, group_std]
      - dose_stats: (mean, std) of log2(dose) computed on train
      - train_stats: dict {(cell_type, name): (mean_vec, std_vec)}
      - global_by_cell: dict {cell_type: (mean_vec, std_vec)} fallback when similars missing
      - cell_type_categories: ordered list of train cell types for consistent context encoding
      - cell_context: description of how cell context is encoded
    """
    import numpy as _np
    import pandas as _pd
    from sklearn.decomposition import PCA as _PCA

    X = _to_dense_float32(adata_train.layers[gene_layer])

    cell_type_series = adata_train.obs[cell_type_col].astype("category")
    cell_types = cell_type_series.astype(str).values
    names = _np.asarray(cond_stage_data_train).astype(str)
    cell_type_categories = list(cell_type_series.cat.categories.astype(str).values)
    stats_space = "absolute"

    rep_mode = str(cell_type_rep).strip().lower()
    need_control = bool(residualize_to_control) or rep_mode == "control_pca"
    control_stats_by_cell = None
    control_count_by_cell = None
    resolved_control_col = None
    resolved_control_label = None
    if need_control:
        control_source = control_adata if control_adata is not None else adata_train
        control_mask, resolved_control_col, resolved_control_label = _resolve_control_mask(
            control_source,
            control_col=control_col,
            control_label=control_label,
        )
        source_cell_types = _np.asarray(
            control_source.obs[cell_type_col].astype(str).values
        )
        X_ctrl = _extract_rows_dense_float32(control_source.layers[gene_layer], control_mask)
        control_stats_by_cell, control_count_by_cell = _compute_control_stats_by_cell(
            X_ctrl,
            source_cell_types[control_mask],
            cell_type_categories,
        )

    # group key and index
    key = _pd.Series(cell_types + "||" + names)
    group_ids, group_index = _np.unique(key.values, return_inverse=True)

    num_groups = len(group_ids)
    num_genes = X.shape[1]
    group_mean = _np.empty((num_groups, num_genes), dtype=_np.float32)
    group_std = _np.empty((num_groups, num_genes), dtype=_np.float32)
    proj_mean = _np.empty((num_groups, num_genes), dtype=_np.float32)
    proj_std = _np.empty((num_groups, num_genes), dtype=_np.float32)

    # also compute per-cell_type global fallback stats
    uniq_cells = _np.unique(cell_types)
    global_by_cell = {}
    for ct in uniq_cells:
        rows_ct = cell_types == ct
        Xct = X[rows_ct]
        if Xct.shape[0] > 1:
            global_by_cell[ct] = (
                Xct.mean(axis=0, dtype=_np.float32),
                Xct.std(axis=0, dtype=_np.float32),
            )
        else:
            # fallback to global over all if extremely scarce
            global_by_cell[ct] = (
                X.mean(axis=0, dtype=_np.float32),
                X.std(axis=0, dtype=_np.float32),
            )

    for gi in range(num_groups):
        rows = group_index == gi
        Xg = X[rows]
        n_group = int(rows.sum())
        if Xg.shape[0] > 1:
            group_mean[gi] = Xg.mean(axis=0, dtype=_np.float32)
            group_std[gi] = Xg.std(axis=0, dtype=_np.float32)
        else:
            # fall back to its cell_type-only global
            ct = group_ids[gi].split("||")[0]
            mu, sd = global_by_cell[ct]
            group_mean[gi] = mu
            group_std[gi] = sd

        if residualize_to_control:
            ct = group_ids[gi].split("||")[0]
            ctrl_mu, ctrl_sd = control_stats_by_cell.get(ct, None) or next(
                iter(control_stats_by_cell.values())
            )
            mu_res = group_mean[gi] - ctrl_mu
            sd_res = _np.log(_np.clip(group_std[gi], 1e-6, None)) - _np.log(
                _np.clip(ctrl_sd, 1e-6, None)
            )
            if residual_shrinkage and residual_shrinkage > 0:
                weight = float(n_group) / float(n_group + residual_shrinkage)
                mu_res = mu_res * weight
                sd_res = sd_res * weight
            proj_mean[gi] = mu_res.astype(_np.float32)
            proj_std[gi] = sd_res.astype(_np.float32)
        else:
            proj_mean[gi] = group_mean[gi]
            proj_std[gi] = group_std[gi]

    stats_groups = _np.concatenate([proj_mean, proj_std], axis=1).astype(_np.float32)
    pca = _PCA(
        n_components=n_components, svd_solver="randomized", random_state=random_state
    ).fit(stats_groups)

    # dose normalization (log2) on train
    dose_raw = adata_train.obs[dose_col].astype(_np.float32).values
    dose_log = _np.log2(_np.clip(dose_raw, 1e-6, None)).astype(_np.float32)
    dose_stats = (float(dose_log.mean()), float(dose_log.std() + 1e-8))

    # build dict of per-(cell_type,name) stats for reuse
    train_stats = {}
    for gi, gid in enumerate(group_ids):
        ct, nm = gid.split("||", 1)
        train_stats[(ct, nm)] = (proj_mean[gi], proj_std[gi])

    if rep_mode == "onehot":
        cell_context = {
            "mode": "onehot",
            "dim": int(len(cell_type_categories)),
            "categories": list(cell_type_categories),
        }
    elif rep_mode == "control_pca":
        cell_context = _fit_cell_type_control_context(
            adata_train,
            cell_types=cell_types,
            cell_type_categories=cell_type_categories,
            cell_type_col=cell_type_col,
            gene_layer=gene_layer,
            control_col=control_col,
            control_label=control_label,
            control_adata=control_adata,
            cell_type_pca_n=cell_type_pca_n,
            random_state=random_state,
        )
    else:
        raise ValueError(
            "cell_type_rep must be one of ['onehot', 'control_pca'], "
            f"got {cell_type_rep}"
        )

    if residualize_to_control:
        zero_vec = _np.zeros(num_genes, dtype=_np.float32)
        default_stats_by_cell = {
            ct: (zero_vec.copy(), zero_vec.copy()) for ct in cell_type_categories
        }
        stats_space = "control_residual"
    else:
        default_stats_by_cell = {
            ct: (mu.copy(), sd.copy()) for ct, (mu, sd) in global_by_cell.items()
        }

    return {
        "pca": pca,
        "dose_stats": dose_stats,
        "train_stats": train_stats,
        "global_by_cell": global_by_cell,
        "default_stats_by_cell": default_stats_by_cell,
        "control_stats_by_cell": control_stats_by_cell,
        "control_count_by_cell": control_count_by_cell,
        "control_col": resolved_control_col,
        "control_label": resolved_control_label,
        "cell_type_categories": cell_type_categories,
        "cell_context": cell_context,
        "cell_context_mode": cell_context["mode"],
        "cell_context_dim": int(cell_context["dim"]),
        "stats_space": stats_space,
        "residualize_to_control": bool(residualize_to_control),
        "residual_shrinkage": float(residual_shrinkage),
        "n_components": n_components,
    }


def build_unseen_enhanced_cov_with_similars(
    adata_meta_like,
    unseen_names,
    similars_map,
    train_fit,
    cell_type_col="cell_type",
    dose_col="dose",
    name_to_smiles=None,
    similars_weight_mode="linear",
    similars_score_threshold=0.0,
):
    """
    Construct enhanced covariates for unseen perturbations using only train-derived stats.

    Parameters
    ----------
    adata_meta_like : AnnData (or object with .obs columns), providing per-sample cell_type and dose
    unseen_names    : array-like of length N, names for unseen perturbations per sample
    similars_map    : dict mapping each unseen pair to either:
                      - legacy list[str] of train pair keys
                      - structured dict with a `selected_drugs` / `candidates` list,
                        where each candidate may include `full_key` and `score`
    train_fit       : dict returned by fit_train_stats_pca()
    name_to_smiles  : dict {drug_name: smiles}, optional. If provided, used to map similar drug names to keys in train_stats.
    similars_weight_mode : how candidate scores are transformed into weights.
                           "linear" keeps the current behavior.
                           "threshold_uniform_kept" drops scores below threshold
                           and uses uniform weights over the remaining candidates.
                           "threshold_ceil_sqrt_ceil" drops scores below threshold,
                           then applies ceil(score) -> sqrt -> ceil before weighting.
    similars_score_threshold : minimum raw score kept by threshold-based modes.
    """
    import numpy as _np
    import pandas as _pd

    pca = train_fit["pca"]
    dose_mean, dose_std = train_fit["dose_stats"]
    train_stats = train_fit["train_stats"]
    global_by_cell = train_fit["global_by_cell"]
    default_stats_by_cell = train_fit.get("default_stats_by_cell", global_by_cell)
    cell_type_categories = train_fit["cell_type_categories"]

    N = len(unseen_names)
    cell_types = _np.asarray(adata_meta_like.obs[cell_type_col].astype(str).values)
    dose_raw = _np.asarray(adata_meta_like.obs[dose_col].astype(_np.float32).values)
    dose_log = _np.log2(_np.clip(dose_raw, 1e-6, None)).astype(_np.float32)
    dose_std_arr = (
        ((dose_log - dose_mean) / dose_std).reshape(-1, 1).astype(_np.float32)
    )

    # build cell-context block using the train-fitted representation
    cell_context = _build_cell_context_matrix(cell_types, train_fit)

    # build reduced stats per sample from similars in train
    reduced_list = []
    missing_similars_logged = set()
    for i in range(N):
        ct = cell_types[i]
        unseen = unseen_names[i]

        ct_unseen = ct + "_" + unseen
        sims = extract_similars_candidates(
            similars_map.get(ct_unseen, []) or [],
            cell_type=ct,
        )
        mu_list, sd_list, weight_list = [], [], []
        for sim in sims:
            s_name = sim.get("drug_name") or _extract_similars_drug_name(
                sim.get("full_key", ""),
                ct,
            )
            if not s_name:
                continue
            
            if name_to_smiles and s_name in name_to_smiles:
                s_key_name = name_to_smiles[s_name]
            else:
                s_key_name = str(s_name)
                
            key = (ct, s_key_name)
            if key in train_stats:
                mu_s, sd_s = train_stats[key]
                mu_list.append(mu_s)
                sd_list.append(sd_s)
                weight_list.append(
                    _transform_similars_score_to_weight(
                        sim.get("score"),
                        mode=similars_weight_mode,
                        threshold=similars_score_threshold,
                    )
                )

            # print(f"Found similar drugs for {ct_unseen}: {s_name}")
        if len(mu_list) == 0:
            mu_f, sd_f = default_stats_by_cell.get(ct, None) or next(
                iter(default_stats_by_cell.values())
            )
            if ct_unseen not in missing_similars_logged:
                if train_fit.get("stats_space") == "control_residual":
                    print(
                        f"No similar drugs found for {ct_unseen}, using control residual prior"
                    )
                else:
                    print(f"No similar drugs found for {ct_unseen}, using global stats")
                missing_similars_logged.add(ct_unseen)

            mu_agg, sd_agg = mu_f, sd_f
        else:
            weights = _np.asarray(weight_list, dtype=_np.float32)
            if weights.size == 0 or (not _np.isfinite(weights).all()) or float(weights.sum()) <= 0:
                mu_f, sd_f = default_stats_by_cell.get(ct, None) or next(
                    iter(default_stats_by_cell.values())
                )
                if ct_unseen not in missing_similars_logged:
                    print(
                        f"All similar-drug weights were zero for {ct_unseen}, "
                        "using fallback prior"
                    )
                    missing_similars_logged.add(ct_unseen)
                mu_agg, sd_agg = mu_f, sd_f
                stats_vec = _np.concatenate([mu_agg, sd_agg], axis=0).astype(_np.float32)
                red = pca.transform(stats_vec.reshape(1, -1)).astype(_np.float32)
                reduced_list.append(red)
                continue
            weights = weights / _np.clip(weights.sum(), 1e-8, None)

            mu_stack = _np.stack(mu_list, axis=0).astype(_np.float32)
            sd_stack = _np.stack(sd_list, axis=0).astype(_np.float32)
            mu_agg = _np.tensordot(weights, mu_stack, axes=(0, 0)).astype(_np.float32)
            sd_agg = _np.tensordot(weights, sd_stack, axes=(0, 0)).astype(_np.float32)
        stats_vec = _np.concatenate([mu_agg, sd_agg], axis=0).astype(_np.float32)
        red = pca.transform(stats_vec.reshape(1, -1)).astype(_np.float32)
        reduced_list.append(red)
    reduced_mat = _np.concatenate(reduced_list, axis=0)

    enhanced_cov = _np.concatenate(
        [cell_context, dose_std_arr, reduced_mat], axis=1
    ).astype(_np.float32)
    return enhanced_cov


def build_train_enhanced_cov(
    adata_train,
    cond_stage_data_train,
    train_fit,
    cell_type_col="cell_type",
    dose_col="dose",
):
    """
    Build enhanced covariates for the training set using train-only stats and PCA
    already returned by fit_train_stats_pca.

    Output shape matches build_unseen_enhanced_cov_with_similars:
      [cell_context, dose_std, pca_k]
    """
    import numpy as _np

    pca = train_fit["pca"]
    dose_mean, dose_std = train_fit["dose_stats"]
    train_stats = train_fit["train_stats"]
    global_by_cell = train_fit["global_by_cell"]
    default_stats_by_cell = train_fit.get("default_stats_by_cell", global_by_cell)

    cell_types = _np.asarray(adata_train.obs[cell_type_col].astype(str).values)
    names = _np.asarray(cond_stage_data_train).astype(str)
    N = len(names)

    # dose standardization with train stats
    dose_raw = _np.asarray(adata_train.obs[dose_col].astype(_np.float32).values)
    dose_log = _np.log2(_np.clip(dose_raw, 1e-6, None)).astype(_np.float32)
    dose_std_arr = (
        ((dose_log - dose_mean) / dose_std).reshape(-1, 1).astype(_np.float32)
    )

    # cell context can be one-hot or control-derived PCA prototypes
    cell_context = _build_cell_context_matrix(cell_types, train_fit)

    # map each (cell_type, name) to its stats, fallback to cell-type global if missing
    reduced_list = []
    for i in range(N):
        ct = cell_types[i]
        nm = names[i]
        key = (ct, nm)
        if key in train_stats:
            mu, sd = train_stats[key]
        else:
            mu, sd = default_stats_by_cell.get(ct, None) or next(
                iter(default_stats_by_cell.values())
            )
        stats_vec = _np.concatenate([mu, sd], axis=0).astype(_np.float32)
        red = pca.transform(stats_vec.reshape(1, -1)).astype(_np.float32)
        reduced_list.append(red)
    reduced_mat = _np.concatenate(reduced_list, axis=0)

    enhanced_cov = _np.concatenate(
        [cell_context, dose_std_arr, reduced_mat], axis=1
    ).astype(_np.float32)
    return enhanced_cov
