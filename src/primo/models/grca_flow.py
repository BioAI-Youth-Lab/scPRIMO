#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.nn as nn

from primo.vendor.perturbnet.net2net.modules.flow.loss import NLL
from primo.vendor.perturbnet.net2net.modules.util import log_txt_as_img
from primo.vendor.perturbnet.net2net.modules.flow.flatflow import *

from primo.vendor.perturbnet.pytorch_scvi.distributions import *
from primo.vendor.perturbnet.pytorch_scvi.scvi_generate_z import *

from primo.utils.perturbation_embeddings import ConcatDatasetWithIndices


class ConcatDatasetWithIndices(torch.utils.data.Dataset):
    """
    Module of concatenating multiple datasets in a data loader
    """

    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple([d[i] for d in self.datasets] + [i])

    def __len__(self):
        return min(len(d) for d in self.datasets)


class Net2NetFlow_TFVAE_Covariate_Flow(nn.Module):
    """
    cINN module connecting standardized ChemicalVAE latent space to VAE/scvi latent space.Hgraph also implemented
    """

    def __init__(
        self,
        configured_flow,
        first_stage_data,
        cond_stage_data,
        perturbToOnehotLib,
        oneHotData,
        model_con,
        std_model,
        sess=None,
        enc_ph=None,
        z_gen_data_v=None,
        is_training=None,
        covariates=None,
        scvi_model=None,
        model_to_use=None,
        ignore_keys=[],
        first_stage_key="cell",
        cond_stage_key="cell",
        interpolate_cond_size=-1,
        chemberta_global=None,
        chemberta_local=None,
        chemberta_name_to_index=None,
        chemberta_cond_stage_data=None,
        chemberta_alpha=1.0,
        chemberta_pca_n=None,
        n_epochs=None,
        cell_type_dim=6,
        drug_proj_mode="none",
        drug_proj_dim=None,
    ):
        super().__init__()
        self.flow = configured_flow
        self.loss = NLL()
        self.first_stage_data = first_stage_data
        self.cond_stage_data = cond_stage_data
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.interpolate_cond_size = interpolate_cond_size
        self.training_time = 0
        self.train_loss = []
        self.test_loss = []
        self.train_flow_loss = []
        self.test_flow_loss = []
        self.train_centroid_loss = []
        self.test_centroid_loss = []
        self.train_candidate_loss = []
        self.test_candidate_loss = []
        self.train_decoded_effect_loss = []
        self.test_decoded_effect_loss = []
        self.perturbToOnehotLib = perturbToOnehotLib
        self.oneHotData = oneHotData
        self.model_con = model_con
        self.std_model = std_model
        self.sess = sess
        self.enc_ph = enc_ph
        self.z_gen_data_v = z_gen_data_v
        self.is_training = is_training
        self.covariates = covariates
        self.scvi_model = scvi_model
        self.chemberta_global = chemberta_global  # Global ChemBERTa embeddings
        self.chemberta_local = chemberta_local  # Local ChemBERTa embeddings (mean)
        self.chemberta_name_to_index = (
            chemberta_name_to_index  # Optional: map perturb name -> chemberta row
        )
        self.chemberta_cond_stage_data = (
            chemberta_cond_stage_data  # Dataset-row-aligned catalog_number array
        )
        self.chemberta_alpha = chemberta_alpha
        # PCA params (optional): set by _fit_chemberta_pca when chemberta_pca_n is not None
        self.chemberta_pca_n = chemberta_pca_n
        self.chemberta_pca_components_ = None
        self.chemberta_pca_mean_ = None
        self._onehot_index_to_name = None  # lazily built reverse map
        self.n_epochs = n_epochs
        self.cell_type_dim = cell_type_dim
        self.drug_proj_mode = self._resolve_drug_proj_mode(drug_proj_mode)
        self.drug_proj_dim = (
            None if drug_proj_dim in (None, "", 0, "0") else int(drug_proj_dim)
        )
        self._drug_proj = None
        self._drug_proj_in_dim = None
        self._drug_proj_out_dim = None
        # FiLM fusion layers for conditioning (drug + cell features), lazily initialized
        # Groups: drug_group (drug SMILES + dose) and cell_group (cell_type + gene_stats)
        # FiLM: drug_group generates scale/shift to modulate cell_group
        self._fuse_initialized = False
        self._fuse_hidden_dim = 512
        self._drug_film_fc1 = None  # Drug -> hidden for FiLM params
        self._drug_film_fc2 = None  # Hidden -> scale & shift
        self._cell_fc = None  # Cell group processing
        self._fuse_fc = None  # Final fusion layer
        # Legacy variables (kept for compatibility)
        self._fuse_fc1 = None
        self._fusion_layer = None
        self._fuse_gate = None
        self._fuse_bias = None
        self._query_proj = None
        self._key_proj = None
        self._value_proj = None
        self._output_proj = None
        # Lightweight simple-fusion layers (for conditional_vector="simple")
        self._simple_fuse_initialized = False
        self._simple_drug_to_film = None
        self._simple_gate = None
        self._simple_bias = None
        self._simple_expected_dims = None
        self.simple_film_strength = 0.1
        self.simple_gate_strength = 0.1
        self.simple_bias_strength = 0.1
        # Anchor-preserving adapter: starts at identity concat and only learns
        # a bounded drug-conditioned update on the leading cell-context block.
        self._adapter_initialized = False
        self._adapter_hidden_dim = 256
        self._adapter_fc1 = None
        self._adapter_fc2 = None
        self._adapter_gate = None
        self._adapter_expected_dims = None
        self.adapter_strength = 0.50
        self.adapter_max_ratio = 0.40
        # GRCA: 只在细胞上下文块上做 gated residual FiLM 调制。
        self._grca_initialized = False
        self._grca_hidden_dim = 256
        self._grca_fc1 = None
        self._grca_film = None
        self._grca_gate = None
        self._grca_expected_dims = None
        self.grca_strength = 0.50
        self.grca_max_ratio = 0.40
        self._init_residual_fusion_layers()
        self._init_anchor_adapter_layers()

        
    def _get_module_device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _get_model_con_device(self):
        try:
            return next(self.model_con.parameters()).device
        except StopIteration:
            return self._get_module_device()

    def _resolve_drug_proj_mode(self, drug_proj_mode):
        mode = str(drug_proj_mode or "none").strip().lower()
        alias = {
            "none": "none",
            "off": "none",
            "false": "none",
            "linear": "linear",
        }
        if mode not in alias:
            raise ValueError(
                "drug_proj_mode must be one of ['none', 'linear']. "
                f"Got: {drug_proj_mode}"
            )
        return alias[mode]

    def _ensure_drug_proj_layer(self, input_dim, device=None):
        input_dim = int(input_dim)
        if self.drug_proj_mode == "none":
            self._drug_proj_in_dim = input_dim
            self._drug_proj_out_dim = input_dim
            return input_dim

        if self.drug_proj_dim is None or int(self.drug_proj_dim) <= 0:
            raise ValueError("drug_proj_mode=linear 时，drug_proj_dim 必须是正整数。")

        out_dim = int(self.drug_proj_dim)
        if (
            self._drug_proj is None
            or self._drug_proj_in_dim != input_dim
            or self._drug_proj_out_dim != out_dim
        ):
            self._drug_proj = nn.Linear(input_dim, out_dim)
            self._drug_proj_in_dim = input_dim
            self._drug_proj_out_dim = out_dim

        if device is not None and self._drug_proj.weight.device != device:
            self._drug_proj = self._drug_proj.to(device)
        return out_dim

    def _project_drug_tensor(self, z):
        if not isinstance(z, torch.Tensor):
            z = torch.as_tensor(z, dtype=torch.float32, device=self._get_module_device())
        if z.ndim != 2:
            raise ValueError(f"drug latent must be 2D, got shape={tuple(z.shape)}")

        self._ensure_drug_proj_layer(z.shape[1], device=z.device)
        if self.drug_proj_mode == "none":
            return z
        return self._drug_proj(z)

    def _prepare_drug_condition(self, batch_cond_x):
        z = self.encode_con(batch_cond_x)
        std_device = getattr(self.std_model, "device", z.device)
        std_device = (
            std_device if isinstance(std_device, torch.device) else torch.device(std_device)
        )
        z = self.std_model.standardize_z_torch(z.to(std_device))
        return self._project_drug_tensor(z)

    @torch.no_grad()
    def project_z_drug_std(self, z_drug_std):
        z = torch.as_tensor(
            z_drug_std, dtype=torch.float32, device=self._get_module_device()
        )
        z = self._project_drug_tensor(z)
        return z.detach().cpu().numpy()


    def _infer_residual_fusion_input_dim(self):
        """
        Infer input dim of:
            concat = [batch_cond_p, batch_cov_x]

        batch_cond_p includes:
        - standardized z from model_con
        - optional ChemBERTa global/local embeddings
        batch_cov_x comes from self.covariates
        """
        if self.covariates is None:
            raise ValueError("self.covariates is None, cannot infer fusion input dim.")

        cov_dim = int(self.covariates.shape[1])

        # 中文说明：
        # 这里走一遍真实药物条件通路，拿到进入 flow 前的最终维度。
        model_device = self._get_model_con_device()

        sample_onehot = torch.as_tensor(
            self.oneHotData[:1], dtype=torch.float32, device=model_device
        )

        with torch.no_grad():
            z = self._prepare_drug_condition(sample_onehot)

        cond_dim = int(z.shape[1])

        if self.chemberta_global is not None:
            cond_dim += int(self.chemberta_global.shape[1])

        if self.chemberta_local is not None:
            cond_dim += int(self.chemberta_local.shape[1])

        return cond_dim + cov_dim


    def _init_residual_fusion_layers(self):
        in_dim = self._infer_residual_fusion_input_dim()
        hidden = self._fuse_hidden_dim

        self._fuse_fc1 = nn.Linear(in_dim, hidden)
        self._fuse_gate = nn.Linear(hidden, in_dim)
        self._fuse_bias = nn.Linear(hidden, in_dim)

        self._fuse_initialized = True

    def _init_simple_fusion_layers(self, input_dim=None, device=None):
        if self.covariates is None:
            raise ValueError("self.covariates is None, cannot init simple fusion layers.")

        cov_dim = int(self.covariates.shape[1])
        cell_type_dim = int(self.cell_type_dim)
        if cov_dim < (cell_type_dim + 1):
            raise ValueError(
                f"covariates dim too small ({cov_dim}); expect at least "
                f"[cell_type({cell_type_dim}), dose]."
            )

        if input_dim is None:
            if self._fuse_initialized and self._fuse_gate is not None:
                input_dim = int(self._fuse_gate.out_features)
            else:
                input_dim = int(self._infer_residual_fusion_input_dim())

        cond_dim = int(input_dim - cov_dim)
        if cond_dim <= 0:
            raise ValueError(
                f"Invalid simple fusion dims: input_dim={input_dim}, cov_dim={cov_dim}"
            )

        # Grouping rule (simple mode):
        # drug_group = [z_drug, dose, gene_stats]
        # cell_group = [cell_type]
        drug_dim = cond_dim + (cov_dim - cell_type_dim)
        cell_dim = cell_type_dim
        out_dim = drug_dim + cell_dim

        if device is None:
            device = self._get_module_device()

        self._simple_drug_to_film = nn.Linear(drug_dim, cell_dim * 2).to(device)
        self._simple_gate = nn.Linear(out_dim, out_dim).to(device)
        self._simple_bias = nn.Linear(out_dim, out_dim).to(device)
        self._simple_expected_dims = (int(drug_dim), int(cell_dim), int(out_dim))
        self._simple_fuse_initialized = True

    def _init_anchor_adapter_layers(self, input_dim=None, device=None):
        if self.covariates is None:
            raise ValueError("self.covariates is None, cannot init anchor adapter.")

        cov_dim = int(self.covariates.shape[1])
        cell_type_dim = int(self.cell_type_dim)
        if cov_dim < (cell_type_dim + 1):
            raise ValueError(
                f"covariates dim too small ({cov_dim}); expect at least "
                f"[cell_type({cell_type_dim}), dose]."
            )

        if input_dim is None:
            if self._fuse_initialized and self._fuse_gate is not None:
                input_dim = int(self._fuse_gate.out_features)
            else:
                input_dim = int(self._infer_residual_fusion_input_dim())

        cond_dim = int(input_dim - cov_dim)
        if cond_dim <= 0:
            raise ValueError(
                f"Invalid adapter dims: input_dim={input_dim}, cov_dim={cov_dim}"
            )

        drug_dim = cond_dim + (cov_dim - cell_type_dim)
        cell_dim = cell_type_dim
        out_dim = cond_dim + cov_dim
        hidden = int(min(512, max(128, drug_dim // 2)))

        if device is None:
            device = self._get_module_device()

        self._adapter_fc1 = nn.Linear(drug_dim, hidden).to(device)
        self._adapter_fc2 = nn.Linear(hidden, cell_dim).to(device)
        self._adapter_gate = nn.Linear(drug_dim, cell_dim).to(device)
        nn.init.zeros_(self._adapter_fc2.weight)
        nn.init.zeros_(self._adapter_fc2.bias)
        nn.init.zeros_(self._adapter_gate.weight)
        nn.init.constant_(self._adapter_gate.bias, -2.0)
        self._adapter_expected_dims = (int(drug_dim), int(cell_dim), int(out_dim))
        self._adapter_initialized = True

    def _init_grca_layers(self, input_dim=None, device=None):
        if self.covariates is None:
            raise ValueError("self.covariates is None, cannot init GRCA.")

        cov_dim = int(self.covariates.shape[1])
        cell_type_dim = int(self.cell_type_dim)
        if cov_dim < (cell_type_dim + 1):
            raise ValueError(
                f"covariates dim too small ({cov_dim}); expect at least "
                f"[cell_type({cell_type_dim}), dose]."
            )

        if input_dim is None:
            if self._fuse_initialized and self._fuse_gate is not None:
                input_dim = int(self._fuse_gate.out_features)
            else:
                input_dim = int(self._infer_residual_fusion_input_dim())

        cond_dim = int(input_dim - cov_dim)
        if cond_dim <= 0:
            raise ValueError(
                f"Invalid GRCA dims: input_dim={input_dim}, cov_dim={cov_dim}"
            )

        drug_dim = cond_dim + (cov_dim - cell_type_dim)
        cell_dim = cell_type_dim
        out_dim = cond_dim + cov_dim
        hidden = int(min(512, max(128, drug_dim // 2)))

        if device is None:
            device = self._get_module_device()

        self._grca_fc1 = nn.Linear(drug_dim, hidden).to(device)
        self._grca_film = nn.Linear(hidden, cell_dim * 2).to(device)
        self._grca_gate = nn.Linear(drug_dim, 1).to(device)
        nn.init.zeros_(self._grca_film.weight)
        nn.init.zeros_(self._grca_film.bias)
        nn.init.zeros_(self._grca_gate.weight)
        nn.init.constant_(self._grca_gate.bias, -2.0)
        self._grca_expected_dims = (int(drug_dim), int(cell_dim), int(out_dim))
        self._grca_initialized = True

    def _split_condition_blocks(self, batch_cond_p, batch_cov_x):
        cov = batch_cov_x.to(batch_cond_p.device)

        cell_type_dim = int(self.cell_type_dim)
        if cov.shape[1] < (cell_type_dim + 1):
            raise ValueError(
                f"covariates dim={cov.shape[1]} is smaller than expected "
                f"cell_type_dim+1={cell_type_dim + 1}"
            )

        cell_context = cov[:, :cell_type_dim]
        drug_side = cov[:, cell_type_dim:]
        drug_group = torch.cat([batch_cond_p, drug_side], dim=1)
        concat = torch.cat([batch_cond_p, cov], dim=1)
        return concat, cell_context, drug_side, drug_group

    def forward(self, x, c):
        zz, logdet = self.flow(x, c)
        return zz, logdet

    @torch.no_grad()
    def sample_conditional(self, c):
        """
        sample zz for c to use
        """
        z = self.flow.sample(c)
        return z

    @torch.no_grad()
    def generate_zprime(self, x, c, cprime):
        """
        generate new x from x (2 dimensional) and c
        """
        zz, _ = self.flow(x, c)
        zprime = self.flow.reverse(zz, cprime)
        return zprime

    @torch.no_grad()
    def generate_zrec(self, x, c):

        zz, _ = self.flow(x, c)
        zrec = self.flow.reverse(zz, c)
        return zrec

    ##########################

    def get_input(self, key, batch, is_conditioning=False):
        x = batch
        if key in ["caption", "text"]:
            x = list(x[0])
        elif key in ["class"]:
            pass
        else:
            if len(x.shape) == 2:
                x = x[..., None, None]
            elif len(x.shape) == 3:
                x = x[..., None]
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            if is_conditioning:
                if self.interpolate_cond_size > -1:
                    x = F.interpolate(
                        x, size=(self.interpolate_cond_size, self.interpolate_cond_size)
                    )
        return x

    def shared_step(
        self, batch_first, batch_cond, batch_idx, split="train", reverse=False
    ):
        """
        compute the loss value in a batch
        """
        x = batch_first  # self.get_input(self.first_stage_key, batch)
        c = batch_cond  # self.get_input(self.cond_stage_key, batch, is_conditioning = True)
        if reverse:
            x = x.squeeze(-1).squeeze(-1)
            c = c.unsqueeze(-1).unsqueeze(-1)
            zz, logdet = self(c, x)
        else:
            zz, logdet = self(x, c)
        loss, log_dict = self.loss(zz, logdet, split=split)
        return loss, log_dict

    # def configure_optimizers(self, lr):
    #     opt = torch.optim.Adam(
    #         (self.flow.parameters()), lr=lr, betas=(0.5, 0.9), amsgrad=True
    #     )
    #     return opt

    def configure_optimizers(
        self, lr, scheduler_type="plateau", patience=10, factor=0.5, weight_decay=0.0
    ):
        opt = torch.optim.Adam(
            (self.parameters()),
            lr=lr,  
            betas=(0.5, 0.9),
            amsgrad=True,
            weight_decay=weight_decay,
        )

        if scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=factor, patience=patience, verbose=True
            )
        elif scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.n_epochs
            )
        else:
            scheduler = None

        return opt, scheduler

    @torch.no_grad()
    def encode(self, x_torch):
        x_np = x_torch.detach().cpu().numpy()
        feed_dict = {self.enc_ph: x_np, self.is_training: False}
        z = self.sess.run(self.z_gen_data_v, feed_dict=feed_dict)
        z_torch = torch.tensor(z).float()
        return z_torch.unsqueeze(-1).unsqueeze(-1)

    @torch.no_grad()
    def encode_con(self, x_con):
        model_device = self._get_model_con_device()
        if x_con.device != model_device:
            x_con = x_con.to(model_device)
        _, _, _, z = self.model_con(x_con)
        return z

    @torch.no_grad()
    def extractOneHot(self, batch_perturb):
        indx = []
        for i in range(batch_perturb.shape[0]):
            indx.append(
                self.perturbToOnehotLib[self.cond_stage_data[batch_perturb[i].item()]]
            )
        return torch.tensor(self.oneHotData[indx])

    @torch.no_grad()
    def extractOneHotUnseen(self, batch_perturb, cond_stage_data_unseen):
        indx = []
        for i in range(batch_perturb.shape[0]):
            indx.append(
                self.perturbToOnehotLib[cond_stage_data_unseen[batch_perturb[i].item()]]
            )
        return torch.tensor(self.oneHotData[indx])

    @torch.no_grad()
    def extractChemBERTa(self, batch_perturb):

        if self.chemberta_global is None and self.chemberta_local is None:
            print("chemberta_global and chemberta_local are None")
            return None, None
        # Build reverse map: onehot index -> perturb name
        if self._onehot_index_to_name is None:
            # perturbToOnehotLib is name -> index; invert it
            self._onehot_index_to_name = {
                v: k for k, v in self.perturbToOnehotLib.items()
            }

        indx = []
        for i in range(batch_perturb.shape[0]):
            indx.append(batch_perturb[i].item())

        source_names = self.cond_stage_data

        pert_names = [source_names[i] for i in indx]

        if self.chemberta_name_to_index is None:
            raise ValueError(
                "chemberta_name_to_index is required to align perturb names to ChemBERTa embeddings"
            )

        missing = [
            name for name in pert_names if name not in self.chemberta_name_to_index
        ]

        if len(missing) > 0:
            raise ValueError(
                f"Found {len(missing)} perturb names not in chemberta_name_to_index, e.g. {missing[:5]}"
            )

        chem_idx = np.array(
            [self.chemberta_name_to_index[name] for name in pert_names], dtype=int
        )

        if self.chemberta_local is None:
            chemberta_global_arr = np.asarray(self.chemberta_global)
            alpha = getattr(self, "chemberta_alpha", 1.0)
            global_emb = torch.tensor(
                chemberta_global_arr[chem_idx], dtype=torch.float32
            )
            if alpha != 1.0:
                global_emb = global_emb * alpha
            return global_emb, None

        chemberta_global_arr = np.asarray(self.chemberta_global)
        chemberta_local_arr = np.asarray(self.chemberta_local)
        global_emb = torch.tensor(chemberta_global_arr[chem_idx], dtype=torch.float32)
        local_emb = torch.tensor(chemberta_local_arr[chem_idx], dtype=torch.float32)
        alpha = getattr(self, "chemberta_alpha", 1.0)
        if alpha != 1.0:
            global_emb = global_emb * alpha
            local_emb = local_emb * alpha

        return global_emb, local_emb

    @torch.no_grad()
    def extractChemBERTaUnseen(self, batch_perturb, cond_stage_data_unseen):
        """
        Extract ChemBERTa global and local embeddings for unseen perturbations
        """
        if not hasattr(self, "chemberta_global_unseen") or not hasattr(
            self, "chemberta_local_unseen"
        ):
            return None, None

        indx = []
        for i in range(batch_perturb.shape[0]):
            indx.append(batch_perturb[i].item())

        global_emb = torch.tensor(self.chemberta_global_unseen[indx])
        local_emb = torch.tensor(self.chemberta_local_unseen[indx])
 
        return global_emb, local_emb

    def train_cinn(
        self,
        n_epochs=400,
        batch_size=128,
        lr=4.5e-6,
        seed=42,
        start_epoch=1,
        auto_save=0,
        auto_save_path="./",
        use_scheduler=True,
        scheduler_type="cosine",
        centroid_lambda=0.0,
        decoded_effect_lambda=0.0,
        decoded_effect_target_size=1e6,
        weight_decay=0.0,
        conditional_vector="none",
        scvi_give_mean=True,
        save_best_checkpoint=True,
        best_checkpoint_name="model_params_best.pt",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        train the net2net model, with train and test validation
        """
        self.model_con.eval()

        if self.scvi_model is None:
            scvi_flag = False
        else:
            scvi_model = self.scvi_model
            scvi_flag = True
            scvi_decoder_helper = scvi_predictive_z(scvi_model)
            scvi_decoder_device = next(scvi_model.model.parameters()).device

        assert self.first_stage_data.shape[0] == self.cond_stage_data.shape[0]

        begin = time.time()
        conditional_vector_mode = self._resolve_conditional_vector_mode(conditional_vector)
        if conditional_vector_mode == "simple":
            # Init simple-fusion parameters before optimizer creation.
            self._init_simple_fusion_layers(device=device)
        elif conditional_vector_mode == "adapter":
            # Init adapter parameters before optimizer creation.
            self._init_anchor_adapter_layers(device=device)
        elif conditional_vector_mode == "grca":
            # Init GRCA parameters before optimizer creation.
            self._init_grca_layers(device=device)

        n = self.first_stage_data.shape[0]
        n_train = int(n * 0.8)
        n_test = n - n_train

        # a random split of train and validation datasets
        random_state = np.random.RandomState(seed=seed)
        permutation = random_state.permutation(n)
        indices_test, indices_train = permutation[:n_test], permutation[n_test:]

        # dose_cell_onehot = np.concatenate((self.dose_onehot, self.cell_type_onehot), axis = 1)

        cond_stage_train = self.cond_stage_data[indices_train]
        cond_stage_test = self.cond_stage_data[indices_test]
        if scvi_flag:
            # 中文说明：
            # 走 scVI latent 时，first_stage_data 只是为了维持旧 DataLoader 接口，
            # 不应该再复制整块表达矩阵。
            first_stage_train = np.zeros((len(indices_train), 1), dtype=np.float32)
            first_stage_test = np.zeros((len(indices_test), 1), dtype=np.float32)
        else:
            first_stage_train = self.first_stage_data[indices_train]
            first_stage_test = self.first_stage_data[indices_test]

        cond_stage_train = torch.tensor(np.array([indices_train]).reshape(-1))
        cond_stage_test = torch.tensor(np.array([indices_test]).reshape(-1))

        # training
        train_loader = torch.utils.data.DataLoader(
            ConcatDatasetWithIndices(first_stage_train, cond_stage_train),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        test_loader = torch.utils.data.DataLoader(
            ConcatDatasetWithIndices(first_stage_test, cond_stage_test),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        if use_scheduler:
            optimizer, scheduler = self.configure_optimizers(
                lr, scheduler_type, weight_decay
            )
        else:
            optimizer = self.configure_optimizers(lr, None, weight_decay)[0]
            scheduler = None

        # 追踪并保存验证集最优 checkpoint
        best_test_loss = float("inf")
        best_epoch = None
        best_checkpoint_path = None
        if save_best_checkpoint:
            save_dir = auto_save_path if auto_save_path else "."
            os.makedirs(save_dir, exist_ok=True)
            best_checkpoint_path = os.path.join(save_dir, best_checkpoint_name)

        for epoch in range(start_epoch, n_epochs + 1):
            train_loss, test_loss = 0, 0
            train_effect_loss_sum, test_effect_loss_sum = 0.0, 0.0

            self.flow.train()
            for batch_idx, (batch_first_x, batch_cond_x, indices_cov_use) in enumerate(
                train_loader
            ):
                if scvi_flag:
                    idx_cov = np.asarray(indices_cov_use, dtype=np.int64)
                    if idx_cov.ndim == 0:
                        idx_cov = idx_cov.reshape(1)
                    idx_global = indices_train[idx_cov]
                    batch_first = (
                        torch.tensor(
                            scvi_model.get_latent_representation(
                                indices=idx_global, give_mean=scvi_give_mean
                            )
                        )
                        .float()
                        .to(device)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                else:
                    batch_first = self.encode(batch_first_x).to(device)

                batch_cond_raw = batch_cond_x
                batch_cond_x = self.extractOneHot(batch_cond_raw).to(device)

                idx_cov = np.asarray(indices_cov_use, dtype=np.int64)
                if idx_cov.ndim == 0:
                    idx_cov = idx_cov.reshape(1)
                batch_cov_x = torch.from_numpy(
                    self.covariates[indices_train[idx_cov]]
                ).float()

                # 中文说明：
                # 训练侧统一走“标准化后再投影”的药物条件路径。
                batch_cond_p = self._prepare_drug_condition(batch_cond_x).to(device)

                # 如果不要batch_cond_p

                # If ChemBERTa provided, append its embeddings to conditioning vector
                if (
                    self.chemberta_global is not None
                    or self.chemberta_local is not None
                ):
                    global_emb, local_emb = self.extractChemBERTa(batch_cond_raw)

                    if local_emb is None:
                        batch_cond_p = torch.cat(
                            [batch_cond_p, global_emb.to(device)], dim=1
                        )

                        # batch_cond_p = global_emb.to(device)
                    else:
                        batch_cond_p = torch.cat(
                            [batch_cond_p, global_emb.to(device), local_emb.to(device)],
                            dim=1,
                        )

                        # batch_cond_p = torch.cat(
                        #     [global_emb.to(device), local_emb.to(device)],
                        #     dim=1,
                        # )

                batch_cond = self._build_batch_condition(
                    batch_cond_p, batch_cov_x, conditional_vector_mode
                )

                loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)

                # Auxiliary centroid alignment
                if centroid_lambda and centroid_lambda > 0:
                    # deterministic reverse: zero latent
                    z0 = torch.zeros_like(batch_first)
                    z_hat = self.flow.reverse(z0, batch_cond).squeeze(-1).squeeze(-1)
                    z_real = batch_first.squeeze(-1).squeeze(-1).detach()

                    labels = batch_cond_raw.detach()
                    uniq = torch.unique(labels)
                    if uniq.numel() > 0:
                        centroid_mse = torch.zeros(1, device=device)
                        for u in uniq:
                            m = labels == u
                            if torch.any(m):
                                mu_pred = z_hat[m].mean(dim=0)
                                mu_real = z_real[m].mean(dim=0)
                                centroid_mse = centroid_mse + torch.mean(
                                    (mu_pred - mu_real) ** 2
                                )
                        centroid_mse = centroid_mse / uniq.numel()
                        loss = loss + centroid_lambda * centroid_mse

                if (
                    scvi_flag
                    and decoded_effect_lambda
                    and decoded_effect_lambda > 0
                ):
                    library_np = scvi_model.get_latent_library_size(
                        indices=idx_global,
                        give_mean=False,
                    )
                    library_t = torch.tensor(library_np).float().to(device)
                    batch_np = (
                        scvi_model.adata.obs["_scvi_batch"]
                        .to_numpy()[idx_global]
                        .astype(np.int64)
                        .reshape(-1, 1)
                    )
                    batch_t = torch.tensor(batch_np).long().to(scvi_decoder_device)
                    z_real_dec = (
                        batch_first.squeeze(-1).squeeze(-1)
                        .detach()
                        .to(scvi_decoder_device)
                    )
                    z_hat_dec = self.flow.reverse(
                        torch.zeros_like(batch_first), batch_cond
                    ).squeeze(-1).squeeze(-1).to(scvi_decoder_device)
                    library_t = library_t.to(scvi_decoder_device)

                    outputs_real = scvi_decoder_helper.decoder_inference(
                        z_real_dec,
                        library_t,
                        batch_index=batch_t,
                        y=None,
                        n_samples=1,
                    )
                    outputs_hat = scvi_decoder_helper.decoder_inference(
                        z_hat_dec,
                        library_t,
                        batch_index=batch_t,
                        y=None,
                        n_samples=1,
                    )

                    px_rate_real = outputs_real["px_rate"]
                    px_rate_hat = outputs_hat["px_rate"]
                    px_rate_real = px_rate_real / px_rate_real.sum(
                        dim=1, keepdim=True
                    ).clamp_min(1e-6) * float(decoded_effect_target_size)
                    px_rate_hat = px_rate_hat / px_rate_hat.sum(
                        dim=1, keepdim=True
                    ).clamp_min(1e-6) * float(decoded_effect_target_size)
                    mag_real = torch.norm(torch.log1p(px_rate_real), p=2, dim=1)
                    mag_hat = torch.norm(torch.log1p(px_rate_hat), p=2, dim=1)
                    decoded_effect_loss = F.smooth_l1_loss(mag_hat, mag_real)
                    loss = loss + float(decoded_effect_lambda) * decoded_effect_loss
                    train_effect_loss_sum += (
                        float(decoded_effect_loss.item()) * batch_first.shape[0]
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch_first.shape[0]

            self.flow.eval()
            for batch_idx, (batch_first_x, batch_cond_x, indices_cov_use) in enumerate(
                test_loader
            ):
                if scvi_flag:
                    idx_cov = np.asarray(indices_cov_use, dtype=np.int64)
                    if idx_cov.ndim == 0:
                        idx_cov = idx_cov.reshape(1)
                    idx_global = indices_test[idx_cov]
                    batch_first = (
                        torch.tensor(
                            scvi_model.get_latent_representation(
                                indices=idx_global, give_mean=scvi_give_mean
                            )
                        )
                        .float()
                        .to(device)
                        .unsqueeze(-1)
                        .unsqueeze(-1)
                    )
                else:
                    batch_first = self.encode(batch_first_x).to(device)

                # Keep raw indices for ChemBERTa lookup; then build one-hot for model_con
                batch_cond_raw = batch_cond_x
                batch_cond_x = self.extractOneHot(batch_cond_raw).to(device)

                idx_cov = np.asarray(indices_cov_use, dtype=np.int64)
                if idx_cov.ndim == 0:
                    idx_cov = idx_cov.reshape(1)
                batch_cov_x = torch.from_numpy(
                    self.covariates[indices_test[idx_cov]]
                ).float()

                batch_cond_p = self._prepare_drug_condition(batch_cond_x).to(device)



                # If ChemBERTa provided, append its embeddings to conditioning vector
                if (
                    self.chemberta_global is not None
                    or self.chemberta_local is not None
                ):
                    global_emb, local_emb = self.extractChemBERTa(batch_cond_raw)

                    if local_emb is None:

                        batch_cond_p = torch.cat(
                            [batch_cond_p, global_emb.to(device)], dim=1
                        )

                        # batch_cond_p = global_emb.to(device)
                    else:
                        batch_cond_p = torch.cat(
                            [batch_cond_p, global_emb.to(device), local_emb.to(device)],
                            dim=1,
                        )

                        # batch_cond_p = torch.cat(
                        #     [global_emb.to(device), local_emb.to(device)],
                        #     dim=1,
                        # )

                batch_cond = self._build_batch_condition(
                    batch_cond_p, batch_cov_x, conditional_vector_mode
                )

                loss, _ = self.shared_step(batch_first, batch_cond, batch_idx)
                if (
                    scvi_flag
                    and decoded_effect_lambda
                    and decoded_effect_lambda > 0
                ):
                    library_np = scvi_model.get_latent_library_size(
                        indices=idx_global,
                        give_mean=False,
                    )
                    library_t = torch.tensor(library_np).float().to(device)
                    batch_np = (
                        scvi_model.adata.obs["_scvi_batch"]
                        .to_numpy()[idx_global]
                        .astype(np.int64)
                        .reshape(-1, 1)
                    )
                    batch_t = torch.tensor(batch_np).long().to(scvi_decoder_device)
                    z_real_dec = (
                        batch_first.squeeze(-1).squeeze(-1)
                        .detach()
                        .to(scvi_decoder_device)
                    )
                    z_hat_dec = self.flow.reverse(
                        torch.zeros_like(batch_first), batch_cond
                    ).squeeze(-1).squeeze(-1).to(scvi_decoder_device)
                    library_t = library_t.to(scvi_decoder_device)
                    outputs_real = scvi_decoder_helper.decoder_inference(
                        z_real_dec,
                        library_t,
                        batch_index=batch_t,
                        y=None,
                        n_samples=1,
                    )
                    outputs_hat = scvi_decoder_helper.decoder_inference(
                        z_hat_dec,
                        library_t,
                        batch_index=batch_t,
                        y=None,
                        n_samples=1,
                    )
                    px_rate_real = outputs_real["px_rate"]
                    px_rate_hat = outputs_hat["px_rate"]
                    px_rate_real = px_rate_real / px_rate_real.sum(
                        dim=1, keepdim=True
                    ).clamp_min(1e-6) * float(decoded_effect_target_size)
                    px_rate_hat = px_rate_hat / px_rate_hat.sum(
                        dim=1, keepdim=True
                    ).clamp_min(1e-6) * float(decoded_effect_target_size)
                    mag_real = torch.norm(torch.log1p(px_rate_real), p=2, dim=1)
                    mag_hat = torch.norm(torch.log1p(px_rate_hat), p=2, dim=1)
                    decoded_effect_loss = F.smooth_l1_loss(mag_hat, mag_real)
                    loss = loss + float(decoded_effect_lambda) * decoded_effect_loss
                    test_effect_loss_sum += (
                        float(decoded_effect_loss.item()) * batch_first.shape[0]
                    )
                test_loss += loss.item() * batch_first.shape[0]

            train_loss /= n_train
            test_loss /= n_test
            if decoded_effect_lambda and decoded_effect_lambda > 0:
                train_effect_loss_sum /= n_train
                test_effect_loss_sum /= n_test
            else:
                train_effect_loss_sum = 0.0
                test_effect_loss_sum = 0.0

            if use_scheduler and scheduler is not None:
                if scheduler_type == "plateau":
                    scheduler.step(test_loss)  # 基于验证损失调整学习率
                else:
                    scheduler.step()  # 其他调度器类型直接执行

                # 获取当前学习率
                current_lr = optimizer.param_groups[0]["lr"]
            else:
                current_lr = lr

            if save_best_checkpoint and test_loss < best_test_loss:
                best_test_loss = test_loss
                best_epoch = epoch
                torch.save(self.state_dict(), best_checkpoint_path)
                print(
                    "[BestCheckpoint] best_epoch=%d best_test_loss=%.6f"
                    % (best_epoch, best_test_loss)
                )

            if auto_save and epoch % auto_save == 0 and epoch < n_epochs:
                auto_save_path_model = os.path.join(
                    auto_save_path, "model_params" + str(epoch) + ".pt"
                )
                if not os.path.exists(auto_save_path):
                    os.makedirs(auto_save_path, exist_ok=False)
                torch.save(self.state_dict(), auto_save_path_model)

            print(
                "[Epoch %d/%d] [Batch %d/%d] [loss: %f/%f] [lr: %f]"
                % (
                    epoch,
                    n_epochs,
                    batch_idx + 1,
                    len(test_loader),
                    train_loss,
                    test_loss,
                    current_lr,
                )
            )

            self.train_loss.append(train_loss)
            self.test_loss.append(test_loss)
            self.train_decoded_effect_loss.append(train_effect_loss_sum)
            self.test_decoded_effect_loss.append(test_effect_loss_sum)

        self.training_time += time.time() - begin
        if save_best_checkpoint and best_epoch is not None:
            print("[BestCheckpoint] saved best checkpoint to: %s" % best_checkpoint_path)


    def build_conditional_vector_simple(self, batch_cond_p, batch_cov_x):
        # Keep simple mode compatible for legacy checkpoints.
        device = batch_cond_p.device
        concat, cell_group, drug_side, drug_group = self._split_condition_blocks(
            batch_cond_p, batch_cov_x
        )

        out_dim = drug_group.shape[1] + cell_group.shape[1]
        expected_dims = (int(drug_group.shape[1]), int(cell_group.shape[1]), int(out_dim))
        if not self._simple_fuse_initialized:
            self._init_simple_fusion_layers(
                input_dim=int(batch_cond_p.shape[1] + batch_cov_x.shape[1]),
                device=device,
            )
        if self._simple_expected_dims != expected_dims:
            if self.training:
                raise ValueError(
                    "Simple fusion dim mismatch during training: "
                    f"expected {self._simple_expected_dims}, got {expected_dims}. "
                    "This would create new parameters not covered by the current optimizer."
                )
            self._init_simple_fusion_layers(
                input_dim=int(batch_cond_p.shape[1] + batch_cov_x.shape[1]),
                device=device,
            )

        film_params = self._simple_drug_to_film(drug_group)
        scale, shift = torch.chunk(film_params, 2, dim=1)
        cell_mod = (
            cell_group * (1.0 + self.simple_film_strength * torch.tanh(scale))
            + self.simple_film_strength * torch.tanh(shift)
        )

        base = torch.cat([batch_cond_p, cell_mod, drug_side], dim=1)
        if base.shape[1] != concat.shape[1]:
            raise ValueError(
                f"Simple fusion output dim mismatch: base={base.shape[1]} vs concat={concat.shape[1]}"
            )

        gate = torch.sigmoid(self._simple_gate(base))
        bias = torch.tanh(self._simple_bias(base))
        delta_raw = (
            self.simple_gate_strength * (base * gate)
            + self.simple_bias_strength * bias
        )

        max_ratio = 0.50
        delta_norm = torch.norm(delta_raw, p=2, dim=1, keepdim=True)
        ref_norm = torch.norm(concat, p=2, dim=1, keepdim=True).clamp_min(1e-6)
        cap_scale = torch.clamp((max_ratio * ref_norm) / (delta_norm + 1e-6), max=1.0)
        delta = delta_raw * cap_scale
        return concat + delta

    def build_conditional_vector_adapter(self, batch_cond_p, batch_cov_x):
        # Anchor-preserving adapter:
        # - start from canonical concat order [z, cell_context, dose+stats]
        # - update only the cell_context block with bounded residual
        # - initialize at identity for stable warm-start
        device = batch_cond_p.device
        concat, cell_context, drug_side, drug_group = self._split_condition_blocks(
            batch_cond_p, batch_cov_x
        )

        expected_dims = (
            int(drug_group.shape[1]),
            int(cell_context.shape[1]),
            int(concat.shape[1]),
        )
        if not self._adapter_initialized:
            self._init_anchor_adapter_layers(
                input_dim=int(batch_cond_p.shape[1] + batch_cov_x.shape[1]),
                device=device,
            )
        if self._adapter_expected_dims != expected_dims:
            if self.training:
                raise ValueError(
                    "Adapter dim mismatch during training: "
                    f"expected {self._adapter_expected_dims}, got {expected_dims}. "
                    "This would create new parameters not covered by the current optimizer."
                )
            self._init_anchor_adapter_layers(
                input_dim=int(batch_cond_p.shape[1] + batch_cov_x.shape[1]),
                device=device,
            )

        h = F.gelu(self._adapter_fc1(drug_group))
        cell_delta_raw = F.gelu(self._adapter_fc2(h))
        gate = torch.sigmoid(self._adapter_gate(drug_group))
        proposal_cell = cell_context + (self.adapter_strength * gate * cell_delta_raw)

        proposal = torch.cat([batch_cond_p, proposal_cell, drug_side], dim=1)
        delta = proposal - concat
        delta_norm = torch.norm(delta, p=2, dim=1, keepdim=True)
        ref_norm = torch.norm(concat, p=2, dim=1, keepdim=True).clamp_min(1e-6)
        cap_scale = torch.clamp(
            (self.adapter_max_ratio * ref_norm) / (delta_norm + 1e-6), max=1.0
        )
        return concat + (delta * cap_scale)

    def build_conditional_vector_grca(self, batch_cond_p, batch_cov_x):
        # GRCA 只更新 cell_context，药物 latent 和 drug_side 永远保持锚点不变。
        device = batch_cond_p.device
        concat, cell_context, drug_side, drug_group = self._split_condition_blocks(
            batch_cond_p, batch_cov_x
        )

        expected_dims = (
            int(drug_group.shape[1]),
            int(cell_context.shape[1]),
            int(concat.shape[1]),
        )
        if not self._grca_initialized:
            self._init_grca_layers(
                input_dim=int(batch_cond_p.shape[1] + batch_cov_x.shape[1]),
                device=device,
            )
        if self._grca_expected_dims != expected_dims:
            if self.training:
                raise ValueError(
                    "GRCA dim mismatch during training: "
                    f"expected {self._grca_expected_dims}, got {expected_dims}. "
                    "This would create new parameters not covered by the current optimizer."
                )
            self._init_grca_layers(
                input_dim=int(batch_cond_p.shape[1] + batch_cov_x.shape[1]),
                device=device,
            )

        h = F.gelu(self._grca_fc1(drug_group))
        film_params = self._grca_film(h)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        gate = torch.sigmoid(self._grca_gate(drug_group))
        delta_cell = self.grca_strength * gate * (
            cell_context * torch.tanh(gamma) + torch.tanh(beta)
        )
        proposal_cell = cell_context + delta_cell

        proposal = torch.cat([batch_cond_p, proposal_cell, drug_side], dim=1)
        delta = proposal - concat
        delta_norm = torch.norm(delta, p=2, dim=1, keepdim=True)
        ref_norm = torch.norm(concat, p=2, dim=1, keepdim=True).clamp_min(1e-6)
        cap_scale = torch.clamp(
            (self.grca_max_ratio * ref_norm) / (delta_norm + 1e-6), max=1.0
        )
        return concat + (delta * cap_scale)

    def _resolve_conditional_vector_mode(self, conditional_vector):
        # Backward compatibility:
        #   True  -> "full"
        #   False -> "none"
        if isinstance(conditional_vector, bool):
            return "full" if conditional_vector else "none"
        mode = str(conditional_vector).strip().lower()
        alias = {
            "full": "full",
            "residual": "full",
            "simple": "simple",
            "adapter": "adapter",
            "grca": "grca",
            "gated_residual_cell_adapter": "grca",
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

    def _build_batch_condition(self, batch_cond_p, batch_cov_x, conditional_vector_mode):
        cov = batch_cov_x.to(batch_cond_p.device)
        if conditional_vector_mode == "full":
            return self.build_conditional_vector(batch_cond_p, cov)
        if conditional_vector_mode == "simple":
            return self.build_conditional_vector_simple(batch_cond_p, cov)
        if conditional_vector_mode == "adapter":
            return self.build_conditional_vector_adapter(batch_cond_p, cov)
        if conditional_vector_mode == "grca":
            return self.build_conditional_vector_grca(batch_cond_p, cov)
        if conditional_vector_mode == "none":
            return torch.cat((batch_cond_p, cov), dim=1)
        raise ValueError(f"Unsupported conditional_vector_mode: {conditional_vector_mode}")

    def build_conditional_vector(self, batch_cond_p, batch_cov_x):
        concat = torch.cat([batch_cond_p, batch_cov_x], dim=1)

        if not self._fuse_initialized:
            raise RuntimeError(
                "Residual fusion layers are not initialized. "
                "Expected eager initialization in __init__."
            )

        expected_dim = self._fuse_gate.out_features
        if concat.shape[1] != expected_dim:
            raise ValueError(
                f"Conditional fusion dim mismatch: got {concat.shape[1]}, "
                f"expected {expected_dim}."
            )

        h = F.gelu(self._fuse_fc1(concat))
        gate = torch.sigmoid(self._fuse_gate(h))
        bias = torch.tanh(self._fuse_bias(h))
        fused = concat * (1.0 + 0.1 * gate) + 0.1 * bias
        return fused

    def save(self, dir_path: str, overwrite: bool = False):
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                "{} already exists, Please provide an unexisting director for saving.".format(
                    dir_path
                )
            )
        model_save_path = os.path.join(dir_path, "model_params.pt")

        torch.save(self.state_dict(), model_save_path)

        np.save(os.path.join(dir_path, "training_time.npy"), self.training_time)
        np.save(os.path.join(dir_path, "train_loss.npy"), self.train_loss)
        np.save(os.path.join(dir_path, "test_loss.npy"), self.test_loss)

        # PCA utilities removed; ChemBERTa is pre-reduced offline

    def load(
        self,
        dir_path: str,
        use_cuda: bool = False,
        save_use_cuda: bool = False,
        use_best_checkpoint: bool = False,
        pochs=None,
    ):

        use_cuda = use_cuda and torch.cuda.is_available()
        device = torch.device("cpu") if use_cuda is False else torch.device("cuda")
        if pochs is None and not use_best_checkpoint:
            model_save_path = os.path.join(dir_path, "model_params.pt")
        elif use_best_checkpoint:
            model_save_path = os.path.join(dir_path, "model_params_best.pt")
        else:
            model_save_path = os.path.join(
                dir_path, "model_params" + str(pochs) + ".pt"
            )

        if use_cuda and save_use_cuda:
            self.load_state_dict(torch.load(model_save_path), strict=False)
            self.to(device)
        elif use_cuda and save_use_cuda is False:
            self.load_state_dict(
                torch.load(model_save_path, map_location="cuda:0"), strict=False
            )
            self.to(device)
        else:
            self.load_state_dict(
                torch.load(model_save_path, map_location=device), strict=False
            )
