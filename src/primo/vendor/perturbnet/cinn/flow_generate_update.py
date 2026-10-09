#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import torch
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt


class SCVIZ_CheckNet2Net:
    """Class to use PerturbNet to predict cellular representations and
    count responses (scVI) from perturbation representations
    """

    def __init__(self, model, device, scvi_model_decode):
        super().__init__()
        self.model = model
        self.device = device
        self.scvi_model_decode = scvi_model_decode

    def _maybe_append_chemberta(
        self, condition_np, pert_onehot=None, pert_names=None, pert_smiles=None
    ):

        if (
            getattr(self.model, "chemberta_global", None) is None
            and getattr(self.model, "chemberta_local", None) is None
        ):
            return condition_np
        n = condition_np.shape[0]
        if pert_names is None and pert_onehot is not None:
            arr = np.asarray(pert_onehot)
            if arr.ndim == 3 and arr.shape[1] == 1:
                arr = arr[:, 0, :]
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            onehot_ids = np.argmax(arr, axis=1).astype(int)

            inv_map = {v: k for k, v in self.model.perturbToOnehotLib.items()}
            pert_names = [inv_map[int(i)] for i in onehot_ids]

        chem_idx = None
        name_map = getattr(self.model, "chemberta_name_to_index", None)
        smile_map = getattr(self.model, "chemberta_smile_to_index", None)
        if pert_names is not None and name_map is not None:
            unknown = [nm for nm in pert_names if nm not in name_map]
            if len(unknown) == 0:
                chem_idx = np.array([name_map[nm] for nm in pert_names], dtype=int)
            elif pert_smiles is not None and smile_map is not None:
                chem_idx = np.array([smile_map[sm] for sm in pert_smiles], dtype=int)
            else:
                raise ValueError(
                    f"chemberta_name_to_index missing {len(unknown)} names, e.g. {unknown[:3]}; pass pert_smiles with chemberta_smile_to_index to resolve"
                )
        elif pert_smiles is not None and smile_map is not None:
            chem_idx = np.array([smile_map[sm] for sm in pert_smiles], dtype=int)
        else:
            return condition_np
        # Gather embeddings (model may have PCA pre-configured in extractChemBERTa)
        emb_list = []
        if getattr(self.model, "chemberta_global", None) is not None:
            emb_list.append(np.asarray(self.model.chemberta_global)[chem_idx])
        if getattr(self.model, "chemberta_local", None) is not None:
            emb_list.append(np.asarray(self.model.chemberta_local)[chem_idx])
        if len(emb_list) == 0:
            return condition_np
        chembert = np.concatenate(emb_list, axis=1)
        # Apply alpha scaling if configured (must match preprocessing)
        alpha = getattr(self.model, "chemberta_alpha", 1.0)
        if alpha != 1.0:
            chembert = chembert * alpha
        # PCA is now performed offline in preprocessing; do not project here
        return np.concatenate([condition_np, chembert], axis=1)

    def _chemberta_from_index(self, n_rows, chemberta_emb_idx=None):
        """
        Build a ChemBERTa embedding matrix for a batch when all rows share the
        same chemical (unseen perturbation). If chemberta is not configured or
        index is None, returns None.
        """
        has_global = getattr(self.model, "chemberta_global", None) is not None
        has_local = getattr(self.model, "chemberta_local", None) is not None
        if chemberta_emb_idx is None or (not has_global and not has_local):
            return None
        idx = int(chemberta_emb_idx)
        emb_list = []
        if has_global:
            g = np.asarray(self.model.chemberta_global)[idx]
            emb_list.append(g.reshape(1, -1))
        if has_local:
            l = np.asarray(self.model.chemberta_local)[idx]
            emb_list.append(l.reshape(1, -1))
        if len(emb_list) == 0:
            return None
        chem = np.concatenate(emb_list, axis=1)
        alpha = getattr(self.model, "chemberta_alpha", 1.0)
        if alpha != 1.0:
            chem = chem * alpha
        # tile to batch
        return np.repeat(chem, n_rows, axis=0)

    def _to_batch_index_np(self, batch_data, n_rows):
        """
        Normalize batch index input to int64 numpy array with shape [B, 1].
        Accepts numpy/torch/list/scalar.
        """
        if batch_data is None:
            return None

        batch_np = (
            batch_data.detach().cpu().numpy()
            if isinstance(batch_data, torch.Tensor)
            else np.asarray(batch_data)
        )

        if batch_np.ndim == 0:
            batch_np = np.full((n_rows, 1), int(batch_np), dtype=np.int64)
        elif batch_np.ndim == 1:
            if batch_np.shape[0] != n_rows:
                raise ValueError(
                    f"batch_data length mismatch: got {batch_np.shape[0]}, expected {n_rows}"
                )
            batch_np = batch_np.reshape(-1, 1)
        elif batch_np.ndim == 2:
            if batch_np.shape[0] != n_rows:
                raise ValueError(
                    f"batch_data rows mismatch: got {batch_np.shape[0]}, expected {n_rows}"
                )
            if batch_np.shape[1] != 1:
                raise ValueError(
                    f"batch_data must have shape [B,1], got {batch_np.shape}"
                )
        else:
            raise ValueError(
                f"batch_data must be scalar/1D/2D, got ndim={batch_np.ndim}"
            )

        return batch_np.astype(np.int64)

    def _get_model_device(self):
        """
        Resolve runtime device from model parameters/buffers.
        Falls back to self.device if model has no parameters.
        """
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            try:
                return next(self.model.buffers()).device
            except StopIteration:
                return self.device

    def _resolve_conditional_vector_mode(self, conditional_vector):
        # Reuse training-side resolver when available.
        if hasattr(self.model, "_resolve_conditional_vector_mode"):
            return self.model._resolve_conditional_vector_mode(conditional_vector)

        if isinstance(conditional_vector, bool):
            return "full" if conditional_vector else "none"
        mode = str(conditional_vector).strip().lower()
        alias = {
            "full": "full",
            "residual": "full",
            "simple": "simple",
            "adapter": "adapter",
            "grca": "grca",
            "none": "none",
            "concat": "none",
            "raw": "none",
            "true": "full",
            "false": "none",
        }
        if mode not in alias:
            raise ValueError(
                "conditional_vector must be one of: "
                "['full', 'simple', 'adapter', 'grca', 'none'] (or bool for compatibility). "
                f"Got: {conditional_vector}"
            )
        return alias[mode]

    @torch.no_grad()
    def sample_data(
        self,
        covariates,
        library_data,
        batch_size=50,
        z_drug_std=None,
        chemberta_emb_idx=None,
        pert_onehot=None,
        pert_names=None,
        pert_smiles=None,
        temperature=1.0,
        cell_type_dim=3,
        deterministic=False,
        batch_data=None,
        strict_cond_alignment=False,
        conditional_vector="none",
    ):
        """
        Sample counts given condition built as [ChemBERTa?, covariates].
        - covariates: numpy/torch array of shape [B, cov_dim] (or None)
        - chemberta_emb_idx: int or None. If provided, use same ChemBERTa row for all B.
          If None, optionally infer from pert_onehot/names/smiles using legacy path.
        - temperature: float, sampling temperature. >1.0 increases variance, <1.0 decreases variance.
        - batch_data: optional batch indices for scVI decoder, shape [B,1] (or [B]). Required when scVI n_batch>1.
        - strict_cond_alignment: if True, error out when model has simple-conditioning API
          but FiLM layers are not loaded/initialized (prevents silent legacy fallback).
        - conditional_vector: 'none' | 'simple' | 'full' (also supports True/False for compatibility).
          Controls how cond vector is built from [z, covariates].
        - cell_type_dim: kept only for backward compatibility. The trained model's
          self.model.cell_type_dim is authoritative when conditioning is built.
        """
        # standardize inputs to numpy
        model_device = self._get_model_device()
        cov_np = None
        model_cell_dim = getattr(self.model, "cell_type_dim", None)
        if model_cell_dim is not None:
            try:
                requested_dim = int(cell_type_dim)
                if requested_dim != int(model_cell_dim) and not getattr(
                    self, "_warned_cell_type_dim_mismatch", False
                ):
                    print(
                        "[WARN] sample_data(cell_type_dim=%d) does not control the "
                        "conditioning split here; using model.cell_type_dim=%d "
                        "from the trained model."
                        % (requested_dim, int(model_cell_dim))
                    )
                    self._warned_cell_type_dim_mismatch = True
            except (TypeError, ValueError):
                pass

        if z_drug_std is not None:
            z_np = (
                z_drug_std.detach().cpu().numpy()
                if isinstance(z_drug_std, torch.Tensor)
                else np.asarray(z_drug_std)
            )
            if hasattr(self.model, "project_z_drug_std"):
                z_np = self.model.project_z_drug_std(z_np)
            n_rows = z_np.shape[0]

        if covariates is not None:
            cov_np = (
                covariates.detach().cpu().numpy()
                if isinstance(covariates, torch.Tensor)
                else np.asarray(covariates)
            )
            n_rows = cov_np.shape[0]

        # build chem from fixed index if provided; otherwise fall back to legacy append
        chem_np = self._chemberta_from_index(n_rows, chemberta_emb_idx)

        if chem_np is not None:
            z_np = np.concatenate([z_np, chem_np], axis=1)  # [z_drug_std, chem]
        else:
            z_np = self._maybe_append_chemberta(
                z_np,
                pert_onehot=pert_onehot,
                pert_names=pert_names,
                pert_smiles=pert_smiles,
            )
        
        cond_np = None
        cond_mode = self._resolve_conditional_vector_mode(conditional_vector)
        has_covariates = cov_np is not None
        has_full_builder = has_covariates and hasattr(
            self.model, "build_conditional_vector"
        )
        has_simple_builder = has_covariates and hasattr(
            self.model, "build_conditional_vector_simple"
        )
        has_unified_builder = has_covariates and hasattr(
            self.model, "_build_batch_condition"
        )

        # none: direct legacy concat path
        if has_covariates and cond_mode == "none":
            cond_np = np.concatenate([z_np, cov_np], axis=1)

        builder_err = None
        if cond_np is None and has_covariates:
            z_tensor = torch.tensor(z_np).float().to(model_device)
            cov_tensor = torch.tensor(cov_np).float().to(model_device)

            # Best path: call training-side unified dispatcher
            if has_unified_builder:
                try:
                    cond_tensor = self.model._build_batch_condition(
                        z_tensor, cov_tensor, cond_mode
                    )
                    cond_np = cond_tensor.cpu().detach().numpy()
                    print(
                        f"use model._build_batch_condition(mode={cond_mode}) to build conditional vector"
                    )
                except Exception as exc:
                    builder_err = exc

            # Fallback for older models without unified dispatcher
            if cond_np is None:
                if cond_mode == "full" and has_full_builder:
                    try:
                        cond_tensor = self.model.build_conditional_vector(
                            z_tensor, cov_tensor
                        )
                        cond_np = cond_tensor.cpu().detach().numpy()
                        print("use model.build_conditional_vector to build conditional vector")
                    except Exception as exc:
                        if builder_err is None:
                            builder_err = exc
                elif cond_mode == "simple" and has_simple_builder:
                    try:
                        cond_tensor = self.model.build_conditional_vector_simple(
                            z_tensor, cov_tensor
                        )
                        cond_np = cond_tensor.cpu().detach().numpy()
                        print(
                            "use model.build_conditional_vector_simple to build conditional vector"
                        )
                    except Exception as exc:
                        if builder_err is None:
                            builder_err = exc

        if cond_np is None:
            if (
                cond_mode != "none"
                and has_covariates
                and strict_cond_alignment
                and (has_unified_builder or has_full_builder or has_simple_builder)
            ):
                raise RuntimeError(
                    "Unable to build conditional vector via model API, and strict_cond_alignment=True. "
                    f"last_builder_error={repr(builder_err)}"
                )

            if has_covariates:
                if cond_mode != "none" and not getattr(
                    self.model, "_warned_legacy_condition_concat", False
                ):
                    print(
                        "[WARN] Falling back to legacy cond concat [z,cov]. "
                        "This may be misaligned with training conditioning path."
                    )
                    setattr(self.model, "_warned_legacy_condition_concat", True)
                cond_np = np.concatenate([z_np, cov_np], axis=1)
            else:
                cond_np = z_np
            
        # cond_np = np.concatenate([z_np, cov_np], axis=1) if cov_np is not None else z_np

        # Apply temperature scaling to sampling
        if temperature != 1.0:
            # Create a custom sample_conditional that applies temperature
            sampled_z = self._sample_with_temperature(cond_np, temperature)
        else:
            # Original sampling without temperature
            sampled_z = (
                self.model.sample_conditional(
                    torch.tensor(cond_np).float().to(model_device)
                )
                .squeeze(-1)
                .squeeze(-1)
                .cpu()
                .detach()
                .numpy()
            )

        n_batch = int(
            getattr(getattr(self.scvi_model_decode.model, "model", None), "n_batch", 1)
        )
        batch_np = self._to_batch_index_np(batch_data, sampled_z.shape[0])

        if batch_np is None and n_batch > 1:
            raise ValueError(
                "SCVI decoder requires real batch indices because n_batch=%d. "
                "Pass batch_data with shape [B,1] (long/int), e.g. adata_subset.obs['_scvi_batch']."
                % n_batch
            )

        if batch_np is not None:
            sampled_data = (
                self.scvi_model_decode.posterior_predictive_sample_from_Z_with_batch(
                    sampled_z,
                    library_data,
                    batch_np,
                    batch_size=batch_size,
                    deterministic=deterministic,
                )
            )
        else:
            sampled_data = self.scvi_model_decode.posterior_predictive_sample_from_Z(
                sampled_z,
                library_data,
                batch_size=batch_size,
                deterministic=deterministic,
            )

        return sampled_z, sampled_data

    @torch.no_grad()
    def _sample_with_temperature(self, cond_np, temperature):
        """
        Sample with enhanced temperature scaling to better control the spread of generated samples.
        Temperature > 1.0 increases variance (more spread out), < 1.0 decreases variance (more concentrated).
        Enhanced implementation to better preserve the temperature effect through the flow transformation.
        """
        # Convert condition to tensor
        model_device = self._get_model_device()
        cond_tensor = torch.tensor(cond_np).float().to(model_device)

        # Ensure cond_tensor has 4 dimensions (batch_size, cond_dim, 1, 1)
        # This matches the expected input shape for ConditionalDoubleVectorCouplingBlock
        if len(cond_tensor.shape) == 2:
            cond_tensor = cond_tensor.unsqueeze(-1).unsqueeze(-1)

        # Get the flow model
        flow = self.model.flow

        # Sample base noise with temperature scaling
        batch_size = cond_tensor.shape[0]
        latent_dim = 1204

        # Enhanced temperature application:
        # 1. First generate standard normal noise
        base_noise = torch.randn(batch_size, latent_dim, device=model_device)

        # 2. Apply stronger temperature scaling - cubic scaling for more pronounced effect
        scaled_noise = (
            base_noise * temperature
        )  # Using squared temperature for stronger effect

        # Ensure scaled_noise has 4 dimensions (batch_size, latent_dim, 1, 1)
        scaled_noise = scaled_noise.unsqueeze(-1).unsqueeze(-1)

        # Use the flow to transform the scaled noise
        sampled_z = flow.reverse(scaled_noise, cond_tensor)

        # Remove spatial dimensions
        if sampled_z.dim() > 2:
            sampled_z = sampled_z.squeeze(-1).squeeze(-1)

        # Additional post-processing temperature adjustment
        # This helps preserve the temperature effect that might be diminished by the flow transformation
        # sampled_z = sampled_z * temperature

        return sampled_z.cpu().detach().numpy()

    @torch.no_grad()
    def recon_v(self, latent, condition):
        recon_val, _ = self.model.flow(
            torch.tensor(latent).float().to(self.device).unsqueeze(-1).unsqueeze(-1),
            torch.tensor(condition).float().to(self.device),
        )
        return recon_val.squeeze(-1).squeeze(-1).cpu().detach().numpy()

    @torch.no_grad()
    def trans_data(
        self,
        latent,
        condition,
        condition_new,
        library_data,
        batch_size=50,
        pert_onehot=None,
        pert_names=None,
        pert_onehot_new=None,
        pert_names_new=None,
        pert_smiles=None,
        pert_smiles_new=None,
        deterministic=False,
    ):
        cond_np = np.asarray(condition)
        cond_np = self._maybe_append_chemberta(
            cond_np,
            pert_onehot=pert_onehot,
            pert_names=pert_names,
            pert_smiles=pert_smiles,
        )
        cond_new_np = np.asarray(condition_new)
        cond_new_np = self._maybe_append_chemberta(
            cond_new_np,
            pert_onehot=pert_onehot_new,
            pert_names=pert_names_new,
            pert_smiles=pert_smiles_new,
        )
        trans_z = (
            self.model.generate_zprime(
                torch.tensor(latent)
                .float()
                .to(self.device)
                .unsqueeze(-1)
                .unsqueeze(-1),
                torch.tensor(cond_np).float().to(self.device),
                torch.tensor(cond_new_np).float().to(self.device),
            )
            .squeeze(-1)
            .squeeze(-1)
            .cpu()
            .detach()
            .numpy()
        )

        trans_data = self.scvi_model_decode.posterior_predictive_sample_from_Z(
            trans_z,
            library_data,
            batch_size=batch_size,
            deterministic=deterministic,
        )

        return trans_z, trans_data

    @torch.no_grad()
    def recon_data(
        self,
        latent,
        condition,
        library_data,
        batch_size=50,
        pert_onehot=None,
        pert_names=None,
        pert_smiles=None,
        deterministic=False,
    ):

        cond_np = np.asarray(condition)
        cond_np = self._maybe_append_chemberta(
            cond_np,
            pert_onehot=pert_onehot,
            pert_names=pert_names,
            pert_smiles=pert_smiles,
        )
        rec_z = (
            self.model.generate_zrec(
                torch.tensor(latent)
                .float()
                .to(self.device)
                .unsqueeze(-1)
                .unsqueeze(-1),
                torch.tensor(cond_np).float().to(self.device),
            )
            .squeeze(-1)
            .squeeze(-1)
            .cpu()
            .detach()
            .numpy()
        )

        rec_data = self.scvi_model_decode.posterior_predictive_sample_from_Z(
            rec_z,
            library_data,
            batch_size=batch_size,
            deterministic=deterministic,
        )

        return rec_z, rec_data

    @torch.no_grad()
    def recon_data_with_y(
        self,
        latent,
        condition,
        library_data,
        y_data,
        batch_size=50,
        deterministic=False,
    ):

        rec_z = (
            self.model.generate_zrec(
                torch.tensor(latent)
                .float()
                .to(self.device)
                .unsqueeze(-1)
                .unsqueeze(-1),
                torch.tensor(condition).float().to(self.device),
            )
            .squeeze(-1)
            .squeeze(-1)
            .cpu()
            .detach()
            .numpy()
        )

        rec_data = self.scvi_model_decode.posterior_predictive_sample_from_Z_with_y(
            rec_z,
            library_data,
            y_data,
            batch_size=batch_size,
            deterministic=deterministic,
        )

        return rec_z, rec_data

    @torch.no_grad()
    def sample_data_with_y(
        self, condition_data, library_data, y_data, batch_size=50, deterministic=False
    ):
        sampled_z = (
            self.model.sample_conditional(
                torch.tensor(condition_data).float().to(self.device)
            )
            .squeeze(-1)
            .squeeze(-1)
            .cpu()
            .detach()
            .numpy()
        )

        sampled_data = self.scvi_model_decode.posterior_predictive_sample_from_Z_with_y(
            sampled_z,
            library_data,
            y_data,
            batch_size=batch_size,
            deterministic=deterministic,
        )

        return sampled_z, sampled_data

    @torch.no_grad()
    def trans_data_with_y(
        self,
        latent,
        condition,
        condition_new,
        library_data,
        y_data,
        batch_size=50,
        deterministic=False,
    ):
        trans_z = (
            self.model.generate_zprime(
                torch.tensor(latent)
                .float()
                .to(self.device)
                .unsqueeze(-1)
                .unsqueeze(-1),
                torch.tensor(condition).float().to(self.device),
                torch.tensor(condition_new).float().to(self.device),
            )
            .squeeze(-1)
            .squeeze(-1)
            .cpu()
            .detach()
            .numpy()
        )

        trans_data = self.scvi_model_decode.posterior_predictive_sample_from_Z_with_y(
            trans_z,
            library_data,
            y_data,
            batch_size=batch_size,
            deterministic=deterministic,
        )

        return trans_z, trans_data

    @torch.no_grad()
    def recon_data_with_batch(
        self,
        latent,
        condition,
        library_data,
        batch_data,
        batch_size=50,
        deterministic=False,
    ):

        rec_z = (
            self.model.generate_zrec(
                torch.tensor(latent)
                .float()
                .to(self.device)
                .unsqueeze(-1)
                .unsqueeze(-1),
                torch.tensor(condition).float().to(self.device),
            )
            .squeeze(-1)
            .squeeze(-1)
            .cpu()
            .detach()
            .numpy()
        )

        rec_data = self.scvi_model_decode.posterior_predictive_sample_from_Z_with_batch(
            rec_z,
            library_data,
            batch_data,
            batch_size=batch_size,
            deterministic=deterministic,
        )

        return rec_z, rec_data

    @torch.no_grad()
    def sample_data_with_batch(
        self,
        condition_data,
        library_data,
        batch_data,
        batch_size=50,
        deterministic=False,
    ):
        sampled_z = (
            self.model.sample_conditional(
                torch.tensor(condition_data).float().to(self.device)
            )
            .squeeze(-1)
            .squeeze(-1)
            .cpu()
            .detach()
            .numpy()
        )

        sampled_data = (
            self.scvi_model_decode.posterior_predictive_sample_from_Z_with_batch(
                sampled_z,
                library_data,
                batch_data,
                batch_size=batch_size,
                deterministic=deterministic,
            )
        )

        return sampled_z, sampled_data

    @torch.no_grad()
    def trans_data_with_batch(
        self,
        latent,
        condition,
        condition_new,
        library_data,
        batch_data,
        batch_size=50,
        deterministic=False,
    ):
        trans_z = (
            self.model.generate_zprime(
                torch.tensor(latent)
                .float()
                .to(self.device)
                .unsqueeze(-1)
                .unsqueeze(-1),
                torch.tensor(condition).float().to(self.device),
                torch.tensor(condition_new).float().to(self.device),
            )
            .squeeze(-1)
            .squeeze(-1)
            .cpu()
            .detach()
            .numpy()
        )

        trans_data = (
            self.scvi_model_decode.posterior_predictive_sample_from_Z_with_batch(
                trans_z,
                library_data,
                batch_data,
                batch_size=batch_size,
                deterministic=deterministic,
            )
        )

        return trans_z, trans_data
