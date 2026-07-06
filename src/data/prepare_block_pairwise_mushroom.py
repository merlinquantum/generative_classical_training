import os

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo


# Parameters

SAVE_PATH = "./mushroom_blocks_m100/"

DATASET_ID = 73                  # UCI Mushroom dataset
M_TOTAL = 100                    # final number of photonic modes
N_SELECTED_COLUMNS = 10          # gives n = 10 photons, one photon per categorical block
N_PHOTONS = N_SELECTED_COLUMNS

N_TRAIN = 5000
N_TEST = 1000
N_TOTAL = N_TRAIN + N_TEST

SEED = 0
MISSING_VALUE_TOKEN = "missing"


# Load Mushroom dataset

def load_mushroom_features(dataset_id: int = DATASET_ID,missing_value_token: str = MISSING_VALUE_TOKEN) -> pd.DataFrame:
    """
    Download the UCI Mushroom dataset and return the categorical features.

    Missing categorical values encoded as "?" are replaced by missing_value_token.
    """
    mushroom = fetch_ucirepo(id=dataset_id)

    X_cat = mushroom.data.features.copy()
    X_cat = X_cat.astype(str).replace("?", missing_value_token)

    return X_cat


# Column selection

def select_categorical_columns(X_cat: pd.DataFrame,n_selected_columns: int = N_SELECTED_COLUMNS) -> tuple[pd.DataFrame, list[str]]:
    """
    Select the first n_selected_columns categorical columns.

    Each selected column will become one one-hot block, hence one photon.
    """
    if n_selected_columns > X_cat.shape[1]:
        raise ValueError(
            f"n_selected_columns={n_selected_columns} is larger than the number "
            f"of available columns={X_cat.shape[1]}."
        )

    selected_columns = list(X_cat.columns[:n_selected_columns])
    X_selected = X_cat[selected_columns]

    return X_selected, selected_columns


# Block one-hot encoding

def one_hot_encode_by_blocks(X_selected: pd.DataFrame, selected_columns: list[str]) -> tuple[np.ndarray, dict[str, slice], pd.DataFrame]:
    """
    One-hot encode each categorical column as a separate block.

    For every row and every selected categorical column, exactly one bit is equal to 1.
    Therefore, each row has fixed Hamming weight equal to len(selected_columns).
    """
    blocks = []
    block_slices = {}
    start = 0

    for col in selected_columns:
        dummies = pd.get_dummies(X_selected[col], prefix=col, dtype=int)

        end = start + dummies.shape[1]
        block_slices[col] = slice(start, end)

        blocks.append(dummies)
        start = end

    X_onehot_df = pd.concat(blocks, axis=1)
    X_onehot = X_onehot_df.to_numpy(dtype=int)

    return X_onehot, block_slices, X_onehot_df


# Padding to the target photonic space

def pad_to_total_modes(X_onehot: np.ndarray, m_total: int = M_TOTAL) -> np.ndarray:
    """
    Pad the active one-hot modes with empty zero modes until the final size m_total.
    """
    m_active = X_onehot.shape[1]

    if m_active > m_total:
        raise ValueError(
            f"Too many active one-hot modes: m_active={m_active} > m_total={m_total}. "
            "Select fewer categorical columns or increase M_TOTAL."
        )

    num_extra_modes = m_total - m_active

    X_padded = np.pad(
        X_onehot,
        pad_width=((0, 0), (0, num_extra_modes)),
        mode="constant",
        constant_values=0,
    )

    return X_padded.astype(int)


# Checks

def check_block_dataset(X_padded: np.ndarray, block_slices: dict[str, slice],n_photons: int, m_total: int) -> None:
    """
    Check that the encoded dataset has the expected photonic structure.
    """
    assert X_padded.ndim == 2
    assert X_padded.shape[1] == m_total
    assert (X_padded.sum(axis=1) == n_photons).all()

    # Exactly one photon per categorical block
    for col, sl in block_slices.items():
        block_weight = X_padded[:, sl].sum(axis=1)
        assert (block_weight == 1).all(), f"Problem in block {col}"

    # Padding modes must be zero
    m_active = max(sl.stop for sl in block_slices.values())
    assert (X_padded[:, m_active:] == 0).all()


# Shuffle / split

def shuffle_and_split(X_padded: np.ndarray, n_train: int = N_TRAIN, n_test: int = N_TEST, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """
    Select n_train + n_test samples, shuffle them, and split into train/test sets.
    """
    n_total = n_train + n_test

    if n_total > len(X_padded):
        raise ValueError(
            f"Requested {n_total} samples, but dataset has only {len(X_padded)} rows."
        )

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_padded))[:n_total]
    X_sub = X_padded[perm]

    rng.shuffle(X_sub)

    X_train = X_sub[:n_train]
    X_test = X_sub[n_train:n_train + n_test]

    return X_train, X_test


# Save metadata

def save_block_metadata(save_path: str, selected_columns: list[str], block_slices: dict[str, slice],X_onehot_df: pd.DataFrame) -> None:
    """
    Save the selected columns and their corresponding active-mode blocks.
    """
    rows = []

    for col in selected_columns:
        sl = block_slices[col]
        categories = [name for name in X_onehot_df.columns[sl]]

        rows.append(
            {
                "column": col,
                "start_mode": sl.start,
                "end_mode_exclusive": sl.stop,
                "block_size": sl.stop - sl.start,
                "dummy_columns": "|".join(categories),
            }
        )

    metadata = pd.DataFrame(rows)
    metadata_file = os.path.join(save_path, "mushroom_blocks_metadata_m100_n10.csv")
    metadata.to_csv(metadata_file, index=False)

    print("Saved metadata:")
    print(metadata_file)


# Main script

if __name__ == "__main__":
    # Load Mushroom categorical features
    X_cat = load_mushroom_features(
        dataset_id=DATASET_ID,
        missing_value_token=MISSING_VALUE_TOKEN,
    )

    print("Raw Mushroom shape:", X_cat.shape)
    print("Columns:")
    print(list(X_cat.columns))

    # Choose categorical columns
    X_selected, selected_columns = select_categorical_columns(
        X_cat=X_cat,
        n_selected_columns=N_SELECTED_COLUMNS,
    )

    print("Selected columns:")
    for col in selected_columns:
        print(f"{col:25s} categories = {X_selected[col].nunique()}")

    # One-hot encode by blocks
    X_onehot, block_slices, X_onehot_df = one_hot_encode_by_blocks(
        X_selected=X_selected,
        selected_columns=selected_columns,
    )

    m_active = X_onehot.shape[1]

    print("X_onehot shape before padding:", X_onehot.shape)
    print("m_active =", m_active)
    print("n / number of photons =", N_PHOTONS)
    print("target m =", M_TOTAL)

    # Pad with empty modes until m = 100
    X_padded = pad_to_total_modes(
        X_onehot=X_onehot,
        m_total=M_TOTAL,
    )

    print("X_padded shape:", X_padded.shape)
    print("Extra empty modes:", M_TOTAL - m_active)
    print("Hamming weights:", np.unique(X_padded.sum(axis=1)))

    # Check dataset
    check_block_dataset(
        X_padded=X_padded,
        block_slices=block_slices,
        n_photons=N_PHOTONS,
        m_total=M_TOTAL,
    )

    print("All checks passed.")
    print("Final dataset lives in m =", M_TOTAL, "and n =", N_PHOTONS)

    # Shuffle and split train/test
    X_train, X_test = shuffle_and_split(
        X_padded=X_padded,
        n_train=N_TRAIN,
        n_test=N_TEST,
        seed=SEED,
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Train Hamming weights:", np.unique(X_train.sum(axis=1)))
    print("Test Hamming weights:", np.unique(X_test.sum(axis=1)))

    # Save
    os.makedirs(SAVE_PATH, exist_ok=True)

    train_file = os.path.join(
        SAVE_PATH,
        f"mushroom_blocks_train_m{M_TOTAL}_n{N_PHOTONS}.csv",
    )
    test_file = os.path.join(
        SAVE_PATH,
        f"mushroom_blocks_test_m{M_TOTAL}_n{N_PHOTONS}.csv",
    )

    np.savetxt(train_file, X_train, delimiter=",", fmt="%d")
    np.savetxt(test_file, X_test, delimiter=",", fmt="%d")

    save_block_metadata(
        save_path=SAVE_PATH,
        selected_columns=selected_columns,
        block_slices=block_slices,
        X_onehot_df=X_onehot_df,
    )

    print("Saved files:")
    print(train_file)
    print(test_file)
