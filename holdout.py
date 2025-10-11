import pandas as pd
import numpy as np


def holdout_split(data, target=None, train_size=0.65, random_state=None):
    """
    Retorna (X_train, X_test, y_train, y_test) se target fornecido,
    caso contrário (X_train, X_test).
    """
    if isinstance(data, (list, np.ndarray)):
        data = np.array(data)
    elif isinstance(data, pd.DataFrame):
        data = data.values

    n = len(data)
    np.random.seed(random_state)
    indices = np.random.permutation(n)

    train_end = int(train_size * n)
    train_idx, test_idx = indices[:train_end], indices[train_end:]

    if target is not None:
        target = np.array(target)
        return data[train_idx], data[test_idx], target[train_idx], target[test_idx]
    else:
        return data[train_idx], data[test_idx]


def holdout_indices(n, train_size=0.65, random_state=None):
    """vai retornar (train_idx, test_idx)"""
    rng = np.random.RandomState(random_state)
    indices = rng.permutation(n)
    train_end = int(train_size * n)

    return indices[:train_end], indices[train_end:]


# -------------------------------
# K-FOLD: índices e splits
# -------------------------------

def k_fold_indices(n, k=5, random_state=None, shuffle=True, stratify=None):
    """
    Gera uma lista de (train_idx, val_idx) para k-fold.
    - n: número total de amostras
    - k: número de folds
    - random_state: semente para reprodutibilidade
    - shuffle: embaralhar antes de dividir (usado quando stratify is None)
    - stratify: array de labels (len == n) para stratified split, ou None
       - se stratify fornecido e sklearn disponível, usa StratifiedKFold
       - se stratify fornecido e sklearn NÃO disponível, usa fallback round-robin
    Retorna: lista de tuples (train_idx, val_idx) com tamanho k
    """
    if k <= 1:
        raise ValueError("k deve ser >= 2")
    if n < k:
        raise ValueError("k não pode ser maior que o número de amostras")

    rng = np.random.RandomState(random_state)

    # Se for stratified e sklearn disponível, use StratifiedKFold
    if stratify is not None:
        stratify = np.array(stratify)
        try:
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=random_state)
            folds = []
            # StratifiedKFold fornece indices diretamente
            all_indices = np.arange(n)
            for train_idx, val_idx in skf.split(all_indices, stratify):
                folds.append((train_idx, val_idx))
            return folds
        except Exception:
            # fallback manual: distribui indices por classe em round-robin
            labels = np.array(stratify)
            unique_labels = np.unique(labels)
            buckets = [[] for _ in range(k)]
            for lab in unique_labels:
                lab_idx = np.where(labels == lab)[0]
                lab_idx = rng.permutation(lab_idx)
                for i, idx in enumerate(lab_idx):
                    buckets[i % k].append(idx)
            folds = []
            all_idx = np.arange(n)
            for i in range(k):
                val_idx = np.array(sorted(buckets[i]))
                train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=True)
                folds.append((train_idx, val_idx))
            return folds
    else:
        # não-stratified
        indices = np.arange(n)
        if shuffle:
            indices = rng.permutation(indices)
        folds = []
        # dividir indices em k pedaços (alguns podem ter 1 elemento a mais)
        fold_sizes = [(n // k) + (1 if i < (n % k) else 0) for i in range(k)]
        current = 0
        val_slices = []
        for fs in fold_sizes:
            start, end = current, current + fs
            val_slices.append(indices[start:end])
            current = end
        all_idx = np.arange(n)
        for i in range(k):
            val_idx = np.array(sorted(val_slices[i]))
            train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=True)
            folds.append((train_idx, val_idx))
        return folds


def k_fold_split(data, target=None, k=5, random_state=None, shuffle=True, stratify=False):
    """
    Gera splits K-fold a partir dos arrays/dataframes:
    - Se target for fornecido, retorna lista de tuples (X_train, X_val, y_train, y_val)
    - Caso contrário, retorna lista de tuples (X_train, X_val)
    Parâmetros:
      - data: array-like (list, np.ndarray ou pandas.DataFrame)
      - target: array-like de rótulos (opcional)
      - k: número de folds
      - random_state, shuffle: para reprodutibilidade
      - stratify: bool. Se True e target fornecido, tenta fazer stratified split.
    """
    # converte data
    is_df = False
    if isinstance(data, (list, np.ndarray)):
        X = np.array(data)
    elif isinstance(data, pd.DataFrame):
        X = data.values
        is_df = True
    else:
        # tenta converter
        X = np.array(data)

    n = len(X)
    if n == 0:
        return []

    stratify_array = None
    if stratify:
        if target is None:
            raise ValueError("Para stratify=True é necessário fornecer target")
        stratify_array = np.array(target)

    folds_idx = k_fold_indices(n, k=k, random_state=random_state, shuffle=shuffle, stratify=stratify_array)

    splits = []
    X_arr = X
    y_arr = np.array(target) if target is not None else None

    for train_idx, val_idx in folds_idx:
        if target is not None:
            splits.append((X_arr[train_idx], X_arr[val_idx], y_arr[train_idx], y_arr[val_idx]))
        else:
            splits.append((X_arr[train_idx], X_arr[val_idx]))
    return splits
