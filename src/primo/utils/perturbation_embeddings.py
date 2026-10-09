from __future__ import annotations

import time

import numpy as np
import torch
from tqdm import tqdm


class ConcatDatasetWithIndices(torch.utils.data.Dataset):
    """
    data structure with sample indices of two datasets
    """

    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple([d[i] for d in self.datasets] + [i])

    def __len__(self):
        return min(len(d) for d in self.datasets)


def smiles_to_hot(smiles, max_len, padding, nchars, char_list=None):
    """
    Converts SMILES strings to a one-hot encoded 3D numpy array representation.

    This function takes a list of SMILES (Simplified Molecular Input Line Entry System)
    strings and converts each character in the SMILES sequence to a one-hot encoded
    vector. The function allows specification of the character set used, the maximum
    sequence length, and the type of padding to ensure consistent length among
    sequences.

    Parameters
    ----------
    smiles : list of str
        A list of SMILES strings to be converted.
    max_len : int
        The maximum length of each SMILES string. Shorter strings will be padded,
        while longer strings will cause errors.
    padding : str
        Specifies the padding strategy. Options are 'left' or 'right', determining
        whether padding is added to the beginning or end of each SMILES string.
    nchars : int
        The number of unique characters in the SMILES alphabet, defining the
        third dimension of the output array.
    char_list : list of str, optional
        A list of unique characters expected in the SMILES strings. If not provided,
        a default character list will be used.

    Returns
    -------
    numpy.ndarray
        A 3D numpy array of shape `(num_smiles, max_len, nchars)`, where each
        entry is a one-hot encoded representation of a SMILES character.
    """

    if char_list == None:
        char_list = [
            "7",
            "6",
            "o",
            "]",
            "3",
            "s",
            "(",
            "-",
            "S",
            "/",
            "B",
            "4",
            "[",
            ")",
            "#",
            "I",
            "l",
            "O",
            "H",
            "c",
            "1",
            "@",
            "=",
            "n",
            "P",
            "8",
            "C",
            "2",
            "F",
            "5",
            "r",
            "N",
            "+",
            "\\",
            " ",
        ]
    char_indices = {}
    for i in range(len(char_list)):
        char_indices[char_list[i]] = i

    smiles = [
        pad_smile(i, max_len, padding) for i in smiles if pad_smile(i, max_len, padding)
    ]

    X = np.zeros((len(smiles), max_len, nchars), dtype=np.float32)

    for i, smile in tqdm(enumerate(smiles)):
        for t, char in enumerate(smile):
            try:
                X[i, t, char_indices[char]] = 1
            except KeyError as e:
                print("ERROR: Check chars file. Bad SMILES:", smile)
                raise e
    return X


def pad_smile(string, max_len, padding="right"):
    """
    Pads a SMILES string to a specified maximum length.

    This function pads a given SMILES string to a fixed length by adding
    spaces either to the right or left, based on the specified padding
    strategy. If no padding is needed (i.e., the string length equals or
    exceeds `max_len`), the original string is returned.

    Parameters
    ----------
    string : str
        The SMILES string to be padded.
    max_len : int
        The maximum length of the output string. If the input string is
        shorter than `max_len`, it will be padded with spaces.
    padding : str, optional
        The padding strategy to use, either 'right', 'left', or 'none'
        (default is 'right'). If 'right', spaces are added to the end
        of the string; if 'left', spaces are added to the beginning;
        if 'none', the original string is returned without any padding.

    Returns
    -------
    str
        The padded SMILES string if `padding` is 'right' or 'left', or
        the original string if `padding` is 'none' or if `string` length
        equals or exceeds `max_len`.
    """

    if len(string) <= max_len:
        if padding == "right":
            return string + " " * (max_len - len(string))
        elif padding == "left":
            return " " * (max_len - len(string)) + string
        elif padding == "none":
            return string


def Seq_to_Embed_ESM(ordered_trt, batch_size, model, alphabet, save_path=None):
    """
    Converts sequences to embeddings using the ESM (Evolutionary Scale Modeling) model.

    This function processes a list of sequences in batches, generating embeddings
    for each sequence using a specified ESM model. The embeddings are calculated
    as the mean of token representations for each sequence, excluding padding
    tokens. Optionally, embeddings can be saved to a specified path.

    Parameters
    ----------
    ordered_trt : list of str
        List of sequences to be embedded in order.
    batch_size : int
        The number of sequences to process in each batch.
    model : torch.nn.Module
        The pre-trained ESM model used to generate embeddings.
    alphabet : ESMAlphabet
        Alphabet instance associated with the ESM model, used for tokenization and
        batch conversion.
    save_path : str, optional
        Path to save the resulting embeddings as a `.npy` file. If not provided,
        embeddings are not saved.

    Returns
    -------
    list of numpy.ndarray
        A list containing the embeddings for each sequence.

    """

    data = []
    count = 1
    batch_converter = alphabet.get_batch_converter()
    for i in ordered_trt:
        data.append((count, i))
        count += 1

    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]
    for j in tqdm(range(len(batches))):
        batch = batches[j]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results["representations"][33]
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            sequence_representations.append(
                token_representations[i, 1 : tokens_len + 1]
                .mean(0)
                .numpy()
                .reshape(1, -1)
            )
    if save_path:
        np.save(save_path, sequence_representations)

    return sequence_representations


def create_train_test_splits_by_key(
    adata, train_ratio, add_key, split_key, control, random_seed=None
):
    """
    Splits the observations in an AnnData object into training and testing sets based on unique values of a specified key,
    with certain control values always included in the training set.

    Parameters:
    adata (AnnData): The AnnData object containing the dataset.
    train_ratio (float): The proportion of unique values to include in the train split (0 < train_ratio < 1), excluding control values.
    add_key (str): The key to be added to adata.obs, where the train/test labels will be stored.
    split_key (str): The key in adata.obs used to determine the unique values for making splits.
    control (list): A list of values from the split_key that should always be included in the training set.
    random_seed (int, optional): The seed for the random number generator for reproducibility. If None, the seed is set based on the current time.

    Returns:
    None: The function adds a new column to adata.obs indicating train/test split.
    """
    if random_seed is None:
        random_seed = int(time.time())
    np.random.seed(random_seed)
    unique_values = adata.obs[split_key].unique()
    non_control_values = [value for value in unique_values if value not in control]
    np.random.shuffle(non_control_values)
    num_train = int(np.floor(train_ratio * len(non_control_values)))
    train_values = set(non_control_values[:num_train])
    train_values.update(control)
    adata.obs[add_key] = adata.obs[split_key].apply(
        lambda x: "train" if x in train_values else "test"
    )


def prepare_embeddings_cinn(adata, perturbation_key, trt_key, embed_key):
    """
    Prepare perturbation embeddings for CINN (Cell Inference of Network Navigation) analysis.

    This function extracts perturbation labels, retrieves corresponding embeddings, and creates a mapping
    between perturbation types and their corresponding embedding indices. It is typically used in the context
    of perturbation experiments, where the goal is to relate perturbations to learned embeddings for further analysis.

    Parameters:
    - adata (AnnData): The AnnData object containing the dataset, where perturbation information is stored in `.obs`
      and embeddings are stored in `.uns`.
    - perturbation_key (str): The key in `adata.obs` that stores the perturbation labels (e.g., gene knockdowns, drug treatments).
    - trt_key (str): The key in `adata.uns` where treatment or perturbation types are stored. This is typically an array
      indicating different treatments corresponding to each observation.
    - embed_key (str): The key in `adata.uns` where the embeddings (e.g., from a dimensionality reduction method like PCA or UMAP)
      corresponding to each treatment are stored.

    """

    perturb_with_onehot = np.array(adata.obs[perturbation_key])
    trt_list = np.unique(perturb_with_onehot)
    embed_idx = []
    for i in range(len(trt_list)):
        trt = trt_list[i]
        idx = np.where(adata.uns[trt_key] == trt)[0][0]
        embed_idx.append(idx)
    embeddings = adata.uns[embed_key][embed_idx]

    perturbToEmbed = {}
    for i in range(trt_list.shape[0]):
        perturbToEmbed[trt_list[i]] = i

    return perturb_with_onehot, embeddings, perturbToEmbed
