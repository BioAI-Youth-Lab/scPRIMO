from __future__ import annotations

import numpy as np
import torch
from anndata import AnnData
from scipy import linalg, sparse, stats
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

try:
    import scvi  # type: ignore
except ImportError:  # pragma: no cover
    scvi = None

try:
    import scanpy as sc  # type: ignore
except ImportError:  # pragma: no cover
    sc = None


def compute_mean_z_r2(
    z_pred: np.ndarray, z_real: np.ndarray, labels: np.ndarray
) -> float:
    """
    Compute R^2 between per-label mean latent vectors of predicted and real sets.
    labels: 1D array of group ids (e.g., catalog_number or perturb/dose/cell-type tuples encoded).
    """
    uniq = np.unique(labels)
    mu_pred = []
    mu_real = []
    for u in uniq:
        idx = np.where(labels == u)[0]
        mu_pred.append(z_pred[idx].mean(axis=0))
        mu_real.append(z_real[idx].mean(axis=0))
    mu_pred = np.stack(mu_pred)
    mu_real = np.stack(mu_real)
    # Flatten feature dimension
    y = mu_real.reshape(-1)
    yhat = mu_pred.reshape(-1)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
    return 1.0 - ss_res / ss_tot


def compute_centroid_metrics(
    z_pred: np.ndarray,
    z_real: np.ndarray,
    labels: np.ndarray,
):
    """
    Per-perturbation centroid evaluation.

    Inputs should be sample-aligned: each row of z_pred/z_real corresponds to the
    same sample id in `labels`.

    Returns
    -------
    dict with keys:
      - per_label_distance: {label: L2 distance between centroids}
      - mean_distance: average of the above distances
      - centroid_r2: R^2 across all centroid vectors (via compute_mean_z_r2)
    """
    uniq = np.unique(labels)
    per_label_distance = {}
    for u in uniq:
        idx = np.where(labels == u)[0]
        if idx.size == 0:
            continue
        mu_pred = z_pred[idx].mean(axis=0)
        mu_real = z_real[idx].mean(axis=0)
        per_label_distance[u] = float(np.linalg.norm(mu_pred - mu_real))

    mean_distance = (
        float(np.nanmean(list(per_label_distance.values())))
        if len(per_label_distance) > 0
        else float("nan")
    )
    centroid_r2 = compute_mean_z_r2(z_pred, z_real, labels)
    return {
        "per_label_distance": per_label_distance,
        "mean_distance": mean_distance,
        "centroid_r2": centroid_r2,
    }


def knn_mixing_rate(
    x_real: np.ndarray,
    x_pred: np.ndarray,
    k: int = 30,
    metric: str = "euclidean",
):
    """
    KNN mixing between real and predicted embeddings.

    For each point, compute the fraction of its k nearest neighbors that come from
    the opposite domain (real vs predicted). Higher is better mixing.

    Returns
    -------
    dict with keys:
      - per_point_mix: (n_total,) opposite-domain ratios for every point
      - mean_mix: overall average
      - mean_mix_real / mean_mix_pred: averages per domain
    """
    from sklearn.neighbors import NearestNeighbors

    x_real = np.asarray(x_real)
    x_pred = np.asarray(x_pred)
    X = np.vstack([x_real, x_pred])
    n_real = x_real.shape[0]
    n_pred = x_pred.shape[0]
    n_total = n_real + n_pred
    domain = np.concatenate([np.zeros(n_real, dtype=int), np.ones(n_pred, dtype=int)])

    n_neighbors = int(min(k + 1, n_total))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
    nbrs.fit(X)
    idx = nbrs.kneighbors(X, return_distance=False)

    per_point_mix = np.zeros(n_total, dtype=float)
    for i in range(n_total):
        neigh = idx[i]
        # remove self if present
        if neigh[0] == i:
            neigh = neigh[1:]
        else:
            neigh = neigh[:k]
        per_point_mix[i] = float((domain[neigh] != domain[i]).mean())

    mean_mix = float(per_point_mix.mean())
    mean_mix_real = float(per_point_mix[:n_real].mean()) if n_real > 0 else float("nan")
    mean_mix_pred = float(per_point_mix[n_real:].mean()) if n_pred > 0 else float("nan")

    return {
        "per_point_mix": per_point_mix,
        "mean_mix": mean_mix,
        "mean_mix_real": mean_mix_real,
        "mean_mix_pred": mean_mix_pred,
    }


def compute_mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float = None) -> float:
    """
    Maximum Mean Discrepancy (RBF kernel). Lower is better.
    gamma defaults to 1 / feature_dim if None.
    """
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]
    if gamma is None:
        gamma = 1.0 / float(d)

    def rbf(a, b):
        a2 = (a * a).sum(1, keepdim=True)
        b2 = (b * b).sum(1, keepdim=True)
        dist2 = a2 - 2 * (a @ b.t()) + b2.t()
        return torch.exp(-gamma * dist2)

    k_xx = rbf(x, x)
    k_yy = rbf(y, y)
    k_xy = rbf(x, y)
    mmd2 = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return float(torch.clamp(mmd2, min=0.0).item())


def compute_fid_like(x: np.ndarray, y: np.ndarray) -> float:
    """
    FID-like distance between two Gaussian fits of features. Lower is better.
    """
    mu_x = x.mean(axis=0)
    mu_y = y.mean(axis=0)
    cov_x = np.cov(x, rowvar=False)
    cov_y = np.cov(y, rowvar=False)
    diff = mu_x - mu_y
    # Numerical stabilization
    cov_x += np.eye(cov_x.shape[0]) * 1e-6
    cov_y += np.eye(cov_y.shape[0]) * 1e-6
    # sqrtm via eig
    evals, evecs = np.linalg.eigh(cov_x @ cov_y)
    evals = np.clip(evals, 0.0, None)
    tr_cov = np.trace(cov_x + cov_y - 2 * (evecs @ np.diag(np.sqrt(evals)) @ evecs.T))
    return float(diff @ diff + tr_cov)


class NormalizedRevisionRSquare:
    """
    A class to calculate the R-squared and Pearson correlation metrics for normalized count data.

    This class provides methods for comparing real and synthetic count data through various
    approaches, using both R-squared and Pearson correlation as evaluation metrics.
    Data normalization is performed by row, followed by log transformation.

    Parameters
    ----------
    largeCountData : numpy.ndarray
        A 2D array of count data used to initialize the column-wise normalization
        parameters.
    targetSize : float, optional
        A scaling factor for row-normalized data (default is 1e4). This factor is applied
        to the normalized data before the log transformation.

    Attributes
    ----------
    targetSize : float
        Scaling factor for normalized data.
    col_mu : numpy.ndarray
        The mean of each column in the normalized `largeCountData`, used for column-wise
        normalization. The column normalization has been removed
    col_std : numpy.ndarray
        The standard deviation of each column in the normalized `largeCountData`, used for
        column-wise normalization. The column normalization has been removed.

    Methods
    -------
    calculate_largeData(largeCountData)
        Initializes the `col_mu` and `col_std` by normalizing `largeCountData` by row sums,
        scaling by `targetSize`, and applying a log transformation.

    calculate_pearson(real_data, fake_data)
        Calculates the Pearson correlation coefficient between the average normalized
        real and fake data. Returns 1.5 if NaN values are encountered.

    calculate_r_square(real_data, fake_data)
        Calculates the R-squared metric between the average normalized real and fake data.
        Returns the R-squared value, the normalized real data, and the normalized predicted data.

    calculate_r_square_real_total(real_data, fake_data)
        Calculates the R-squared metric using real data as the true values for comparison.
        Returns the R-squared value, normalized real data, and normalized predicted data.

    calculate_r_square_norm_by_all(real_data, fake_data, norm_vec_real, norm_vec_fake)
        Calculates the R-squared metric by normalizing real and predicted data using external
        normalization vectors. Returns the R-squared value. Applied when the all genes are
        available.

    calculate_pearson_norm_by_all(real_data, fake_data, norm_vec_real, norm_vec_fake)
        Calculates the Pearson correlation coefficient by normalizing real and predicted data
        using external normalization vectors. Returns the Pearson correlation coefficient.
        Applied when the all genes are available.
    """

    def __init__(self, largeCountData, targetSize=1e4):

        self.targetSize = targetSize
        self.calculate_largeData(largeCountData)

    def calculate_largeData(self, largeCountData):
        usedata = largeCountData.copy()
        usedata = usedata / usedata.sum(axis=1)[:, None] * self.targetSize
        usedata = np.log1p(usedata)

        self.col_mu = usedata.mean(axis=0)
        self.col_std = usedata.std(axis=0)

    def calculate_pearson(self, real_data, fake_data):
        real_data_norm = real_data.copy()
        real_data_norm_sum = real_data_norm.sum(axis=1)
        real_data_norm = real_data_norm[real_data_norm_sum != 0, :]

        fake_data_norm = fake_data.copy()
        fake_data_norm_sum = fake_data_norm.sum(axis=1)
        fake_data_norm = fake_data_norm[fake_data_norm_sum != 0, :]

        real_norm_sums = real_data_norm.sum(axis=1)
        real_norm_sums[real_norm_sums == 0] = 1.0
        real_data_norm = real_data_norm / real_norm_sums[:, None] * self.targetSize

        fake_norm_sums = fake_data_norm.sum(axis=1)
        fake_norm_sums[fake_norm_sums == 0] = 1.0
        fake_data_norm = fake_data_norm / fake_norm_sums[:, None] * self.targetSize
        real_data_norm, fake_data_norm = np.log1p(real_data_norm), np.log1p(
            fake_data_norm
        )
        x = np.average(real_data_norm, axis=0)
        y = np.average(fake_data_norm, axis=0)

        # Avoid division by zero / constant input for Pearson
        if np.std(x) == 0:
            x[0] += 1e-6

        if np.isnan(x).any() or np.isnan(y).any():
            return np.nan
        else:
            m, b, r_value, p_value, std_err = stats.linregress(x, y)

            return r_value

    def calculate_r_square(self, real_data, fake_data):
        real_data_norm = real_data.copy()
        real_data_norm_sum = real_data_norm.sum(axis=1)
        real_data_norm = real_data_norm[real_data_norm_sum != 0, :]

        fake_data_norm = fake_data.copy()
        fake_data_norm_sum = fake_data_norm.sum(axis=1)
        fake_data_norm = fake_data_norm[fake_data_norm_sum != 0, :]

        real_norm_sums = real_data_norm.sum(axis=1)
        real_norm_sums[real_norm_sums == 0] = 1.0
        real_data_norm = real_data_norm / real_norm_sums[:, None] * self.targetSize

        fake_norm_sums = fake_data_norm.sum(axis=1)
        fake_norm_sums[fake_norm_sums == 0] = 1.0
        fake_data_norm = fake_data_norm / fake_norm_sums[:, None] * self.targetSize
        real_data_norm, fake_data_norm = np.log1p(real_data_norm), np.log1p(
            fake_data_norm
        )
        x = np.average(real_data_norm, axis=0)
        y = np.average(fake_data_norm, axis=0)

        # Avoid NaN r2 score if data is constant (e.g. all zeros)
        if np.std(x) == 0:
            x[0] += 1e-6

        if np.isnan(x).any() or np.isnan(y).any():
            return np.nan, np.nan, np.nan
        else:
            r2_value = r2_score(x, y)

            return r2_value, real_data_norm, fake_data_norm

    def calculate_r_square_real_total(self, real_data, fake_data):
        real_data_norm = real_data.copy()
        real_data_norm_sum = real_data_norm.sum(axis=1)
        real_data_norm = real_data_norm[real_data_norm_sum != 0, :]

        fake_data_norm = fake_data.copy()
        fake_data_norm_sum = fake_data_norm.sum(axis=1)
        fake_data_norm = fake_data_norm[fake_data_norm_sum != 0, :]

        real_data_norm = (
            real_data_norm / real_data_norm.sum(axis=1)[:, None] * self.targetSize
        )
        fake_data_norm = (
            fake_data_norm / fake_data_norm.sum(axis=1)[:, None] * self.targetSize
        )
        real_data_norm, fake_data_norm = np.log1p(real_data_norm), np.log1p(
            fake_data_norm
        )
        # important to make sure x is y_true and y is y_pred since it will affect which y_bar to be used
        x = np.average(real_data_norm, axis=0)
        y = np.average(fake_data_norm, axis=0)
        r2_value = r2_score(x, y)

        return r2_value, real_data_norm, fake_data_norm

    def calculate_r_square_norm_by_all(
        self, real_data, fake_data, norm_vec_real, norm_vec_fake
    ):
        real_data_norm = real_data.copy()
        real_data_norm_sum = norm_vec_real
        real_data_norm = real_data_norm[real_data_norm_sum != 0, :]

        fake_data_norm = fake_data.copy()
        fake_data_norm_sum = norm_vec_fake
        fake_data_norm = fake_data_norm[fake_data_norm_sum != 0, :]

        real_data_norm = real_data_norm / real_data_norm_sum[:, None] * self.targetSize
        fake_data_norm = fake_data_norm / fake_data_norm_sum[:, None] * self.targetSize
        real_data_norm, fake_data_norm = np.log1p(real_data_norm), np.log1p(
            fake_data_norm
        )
        x = np.average(real_data_norm, axis=0)
        y = np.average(fake_data_norm, axis=0)
        r2_value = r2_score(x, y)
        return r2_value

    def calculate_pearson_norm_by_all(
        self, real_data, fake_data, norm_vec_real, norm_vec_fake
    ):
        real_data_norm = real_data.copy()  
        real_data_norm_sum = norm_vec_real
        real_data_norm = real_data_norm[real_data_norm_sum != 0, :]

        fake_data_norm = fake_data.copy()
        fake_data_norm_sum = norm_vec_fake
        fake_data_norm = fake_data_norm[fake_data_norm_sum != 0, :]

        real_data_norm = real_data_norm / real_data_norm_sum[:, None] * self.targetSize
        fake_data_norm = fake_data_norm / fake_data_norm_sum[:, None] * self.targetSize
        real_data_norm, fake_data_norm = np.log1p(real_data_norm), np.log1p(
            fake_data_norm
        )
        x = np.average(real_data_norm, axis=0)
        y = np.average(fake_data_norm, axis=0)
        try:
            m, b, r_value, p_value, std_err = stats.linregress(x, y)
        except ValueError:
            r_value = np.nan
        return r_value

    def calculate_mse(self, real_data, fake_data):
        real_data_norm = np.asarray(real_data, dtype=np.float64).copy()
        fake_data_norm = np.asarray(fake_data, dtype=np.float64).copy()

        if real_data_norm.ndim == 1:
            real_data_norm = real_data_norm.reshape(1, -1)
        if fake_data_norm.ndim == 1:
            fake_data_norm = fake_data_norm.reshape(1, -1)

        # For MSE we keep zero-sum DEG subsets as valid all-zero vectors instead of
        # dropping them entirely. This avoids NaN outputs when the DEG slice exists
        # but is completely zero in the current evaluation gene space.
        real_sums = real_data_norm.sum(axis=1, keepdims=True)
        fake_sums = fake_data_norm.sum(axis=1, keepdims=True)
        real_sums[real_sums == 0] = 1.0
        fake_sums[fake_sums == 0] = 1.0

        real_data_norm = real_data_norm / real_sums * self.targetSize
        fake_data_norm = fake_data_norm / fake_sums * self.targetSize
        real_data_norm = np.log1p(real_data_norm)
        fake_data_norm = np.log1p(fake_data_norm)

        x = np.average(real_data_norm, axis=0)
        y = np.average(fake_data_norm, axis=0)

        # Mean Squared Error
        mse_value = np.mean((x - y) ** 2)
        return mse_value


class fidscore:
    """
    A class to calculate the Frechet Inception Distance (FID) score and additional
    metrics for comparing distributions of real and predicted data representations.

    This class provides methods to compute the FID score, which measures the similarity
    between two distributions (real and generated data) based on their statistics
    in a lower-dimensional space (e.g., after PCA transformation). However, because
    our adjusted FID has limitation to compare models using different latent space,we don't
    use this metirc anymore。

    Instead, R-squared and Pearson correlation metrics can be computed for a more detailed
    comparison.

    Attributes
    ----------
    pca_50 : sklearn.decomposition.PCA
        PCA transformer set to reduce data dimensionality to 50 components, with a
        random seed for reproducibility.

    Methods
    -------
    calculate_statistics(numpy_data)
        Calculates the mean and covariance matrix of the given data.

    calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6)
        Computes the Frechet Distance between two distributions specified by their
        mean and covariance matrices. Returns both the FID score and an error flag
        for complex matrix values.

    calculate_fid_score(real_data, fake_data, pca_data_fit=None, if_dataPC=False)
        Computes the FID score between the distributions of real and fake data
        by applying PCA transformation and calculating the Frechet Distance in
        the reduced space.

    calculate_r_square(real_data, fake_data)
        Calculates the R-squared metric between the average values of real and
        predicted data. Applied for normalized data.

    calculate_pearson(real_data, fake_data)
        Calculates the Pearson correlation coefficient between the average values
        of real and predicted data distributions. Applied for normalized data.

    """

    def __init__(self):
        super().__init__()
        self.pca_50 = PCA(n_components=50, random_state=42)

    def calculate_statistics(self, numpy_data):
        mu = np.nanmean(numpy_data, axis=0)
        sigma = np.cov(numpy_data, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates" % eps
            )
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                # raise ValueError('Cell component {}'.format(m))
            covmean = covmean.real

            image_error = 1
        else:
            image_error = 0

        tr_covmean = np.trace(covmean)

        return (
            diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        ), image_error

    def calculate_fid_score(
        self, real_data, fake_data, pca_data_fit=None, if_dataPC=False
    ):

        if real_data.shape[0] <= 1 or fake_data.shape[0] <= 1:
            return -1, 1

        all_data = np.concatenate([fake_data, real_data], axis=0)

        if if_dataPC:
            pca_all = pca_data_fit.transform(all_data)
        else:
            pca_all = self.pca_50.fit(real_data).transform(all_data)
        data1, data2 = pca_all[fake_data.shape[0] :], pca_all[: fake_data.shape[0]]

        m1, s1 = self.calculate_statistics(data1)
        m2, s2 = self.calculate_statistics(data2)
        fid_value, image_error = self.calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value, image_error

    def calculate_r_square(self, real_data, fake_data):
        real_data_sum = real_data.sum(axis=1)
        real_data = real_data[real_data_sum != 0, :]
        fake_data_sum = fake_data.sum(axis=1)
        fake_data = fake_data[fake_data_sum != 0, :]
        x = np.average(real_data, axis=0)
        y = np.average(fake_data, axis=0)
        r2_value = r2_score(x, y)

        return r2_value

    def calculate_pearson(self, real_data, fake_data):
        real_data_sum = real_data.sum(axis=1)
        real_data = real_data[real_data_sum != 0, :]
        fake_data_sum = fake_data.sum(axis=1)
        fake_data = fake_data[fake_data_sum != 0, :]
        x = np.average(fake_data, axis=0)
        y = np.average(real_data, axis=0)
        m, b, r_value, p_value, std_err = stats.linregress(x, y)

        return r_value


class fidscore_scvi_extend(fidscore):
    """
    calculate FID score metric defined on the scVI latent space
    """

    def __init__(self, scvi_model):
        super().__init__()

        self.scvi_model = scvi_model

    def calculate_fid_scvi_score(self, real_data, fake_data, give_mean=False):

        if real_data.shape[0] <= 1 or fake_data.shape[0] <= 1:
            return -1, 1

        all_data = np.concatenate([fake_data, real_data], axis=0)
        all_adata = AnnData(X=all_data)
        all_adata.layers["counts"] = all_adata.X.copy()

        repre_all = self.scvi_model.get_latent_representation(
            adata=all_adata, give_mean=give_mean
        )

        data1, data2 = repre_all[fake_data.shape[0] :], repre_all[: fake_data.shape[0]]

        m1, s1 = self.calculate_statistics(data1)
        m2, s2 = self.calculate_statistics(data2)
        fid_value, image_error = self.calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value, image_error

    def calculate_fid_scvi_score_with_y(
        self, real_data, fake_data, y_data, y_key, scvi, adata, give_mean=False
    ):

        if real_data.shape[0] <= 1 or fake_data.shape[0] <= 1:
            return -1, 1

        all_data = np.concatenate([fake_data, real_data], axis=0)
        all_adata = AnnData(X=all_data)
        all_adata.layers["counts"] = all_adata.X.copy()
        all_adata.obs[y_key] = np.concatenate((y_data, y_data))
        # scvi.data.setup_anndata(all_adata, layer = "counts", batch_key = y_key)
        scvi.data.transfer_anndata_setup(adata, all_adata)

        repre_all = self.scvi_model.get_latent_representation(
            adata=all_adata, give_mean=give_mean
        )

        data1, data2 = repre_all[fake_data.shape[0] :], repre_all[: fake_data.shape[0]]

        m1, s1 = self.calculate_statistics(data1)
        m2, s2 = self.calculate_statistics(data2)
        fid_value, image_error = self.calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value, image_error

    def calculate_fid_scvi_indice_score(
        self, real_indices, fake_indices, give_mean=False
    ):

        if len(real_indices) <= 1 or len(fake_indices) <= 1:
            return -1, 1

        n_fake = len(fake_indices)
        all_indices = np.concatenate([np.array(fake_indices), np.array(real_indices)])
        repre_all = self.scvi_model.get_latent_representation(
            indices=all_indices, give_mean=give_mean
        )

        data1, data2 = repre_all[n_fake:], repre_all[:n_fake]

        m1, s1 = self.calculate_statistics(data1)
        m2, s2 = self.calculate_statistics(data2)
        fid_value, image_error = self.calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value, image_error


class fidscore_scgen_extend(fidscore):
    """
    calculate FID score metric defined on the scGen latent space
    """

    def __init__(self, scgen_model):
        super().__init__()
        self.scgen_model = scgen_model

    def calculate_fid_scgen_score(
        self, real_data, fake_data, if_count_layer=False, give_mean=False
    ):
        if real_data.shape[0] <= 1 or fake_data.shape[0] <= 1:
            return -1, 1

        all_data = np.concatenate([fake_data, real_data], axis=0)
        all_adata = AnnData(X=all_data)

        if if_count_layer:
            all_adata.layers["counts"] = all_adata.X.copy()

        repre_all = self.scgen_model.get_latent_representation(
            adata=all_adata, give_mean=give_mean
        )

        data1, data2 = repre_all[fake_data.shape[0] :], repre_all[: fake_data.shape[0]]

        m1, s1 = self.calculate_statistics(data1)
        m2, s2 = self.calculate_statistics(data2)
        fid_value, image_error = self.calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value, image_error


class fidscore_vae_extend(fidscore):
    """
    calculate FID score metric defined on the regular VAE latent space
    """

    def __init__(self, sess, z_gen_data_v, z_gen_mean_v, X_v, is_training):
        super().__init__()

        self.sess = sess
        self.z_gen_data_v = z_gen_data_v
        self.X_v = X_v
        self.z_gen_mean_v = z_gen_mean_v
        self.is_training = is_training

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates" % eps
            )
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))

                # raise ValueError('Cell component {}'.format(m))
            covmean = covmean.real

            image_error = 1
        else:
            image_error = 0

        tr_covmean = np.trace(covmean)

        return (
            diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        ), image_error

    def calculate_fid_vae_score(self, real_data, fake_data, give_mean=False):

        if real_data.shape[0] <= 1 or fake_data.shape[0] <= 1:
            return -1, 1

        all_data = np.concatenate([fake_data, real_data], axis=0)
        # all_adata = AnnData(X = all_data)
        # all_adata.layers["counts"] = all_adata.X.copy()

        # repre_all = self.scvi_model.get_latent_representation(adata = all_adata, give_mean = give_mean)
        feed_dict = {self.X_v: all_data, self.is_training: False}

        if give_mean:
            repre_all = self.sess.run(self.z_gen_mean_v, feed_dict=feed_dict)
        else:
            repre_all = self.sess.run(self.z_gen_data_v, feed_dict=feed_dict)

        data1, data2 = repre_all[fake_data.shape[0] :], repre_all[: fake_data.shape[0]]

        m1, s1 = self.calculate_statistics(data1)
        m2, s2 = self.calculate_statistics(data2)
        fid_value, image_error = self.calculate_frechet_distance(m1, s1, m2, s2)

        return fid_value, image_error
