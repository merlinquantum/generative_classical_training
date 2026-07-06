import os
from itertools import combinations

import numpy as np


# Parameters

FILE_PATH = "./preference_ranking/sushi3a.5000.10.order"
SAVE_PATH = "./preference_ranking/"

N_ITEMS = 10
N_PAIRS_USED = 10             # gives n = 10 photons
M_ACTIVE = 2 * N_PAIRS_USED   # 20 active modes
M_TOTAL = 100                 # final number of modes
N_PHOTONS = N_PAIRS_USED      # one photon per pair

N_TRAIN = 3000
N_TEST = 2000
SEED = 42


# Load sushi rankings

def load_sushi3a_order(file_path: str, n_items: int = N_ITEMS) -> np.ndarray:
    """
    Load sushi3a.5000.10.order.

    Each data line has the format:
        0 10 item_1 item_2 ... item_10

    The first item is the most preferred one.
    """
    rankings = []

    with open(file_path, "r") as f:
        lines = f.readlines()

    # First line is the header: "10 1"
    for line in lines[1:]:
        parts = line.strip().split()

        if len(parts) == 0:
            continue

        length_order = int(parts[1])
        ranking = list(map(int, parts[2:2 + length_order]))

        if len(ranking) != n_items:
            raise ValueError(f"Unexpected ranking length: {len(ranking)}")

        rankings.append(ranking)

    return np.array(rankings, dtype=int)


# Pairwise encoding

def choose_pairs(n_items: int, n_pairs_used: int, seed: int) -> list[tuple[int, int]]:
    """
    Randomly choose n_pairs_used pairwise comparisons among all item pairs.
    """
    all_pairs = list(combinations(range(n_items), 2))

    if n_pairs_used > len(all_pairs):
        raise ValueError(
            f"n_pairs_used={n_pairs_used} is larger than the total number "
            f"of possible pairs C({n_items}, 2)={len(all_pairs)}."
        )

    rng = np.random.default_rng(seed)
    chosen_pair_indices = rng.choice(len(all_pairs), size=n_pairs_used, replace=False)

    return [all_pairs[i] for i in chosen_pair_indices]


def ranking_to_position_map(ranking: np.ndarray) -> dict[int, int]:
    """
    Return a dictionary item -> position in the ranking.
    A smaller position means a more preferred item.
    """
    return {item: pos for pos, item in enumerate(ranking)}


def encode_ranking_pairwise_blocks(ranking: np.ndarray,chosen_pairs: list[tuple[int, int]],m_total: int) -> np.ndarray:
    """
    Encode one ranking into a bitstring of length m_total.

    For each pair (a, b):
        if a is preferred to b -> block 10
        if b is preferred to a -> block 01

    The remaining modes are padded with zeros.
    """
    pos = ranking_to_position_map(ranking)
    bits_active = []

    for a, b in chosen_pairs:
        if pos[a] < pos[b]:
            bits_active.extend([1, 0])
        else:
            bits_active.extend([0, 1])

    bits_active = np.array(bits_active, dtype=int)

    if len(bits_active) > m_total:
        raise ValueError(
            f"m_total={m_total} is too small. It must be at least "
            f"2 * len(chosen_pairs)={2 * len(chosen_pairs)}."
        )

    padding = np.zeros(m_total - len(bits_active), dtype=int)
    return np.concatenate([bits_active, padding])


def encode_rankings_pairwise_blocks(rankings: np.ndarray,chosen_pairs: list[tuple[int, int]],m_total: int) -> np.ndarray:
    """
    Encode all rankings into fixed-Hamming-weight bitstrings.
    """
    return np.array(
        [
            encode_ranking_pairwise_blocks(
                ranking=ranking,
                chosen_pairs=chosen_pairs,
                m_total=m_total,
            )
            for ranking in rankings
        ],
        dtype=int,
    )


# Checks

def check_dataset(bitstrings_arr: np.ndarray,n_samples: int,m_total: int,m_active: int,n_pairs_used: int,n_photons: int) -> None:
    """
    Check that the encoded dataset has the expected photonic structure.
    """
    assert bitstrings_arr.shape == (n_samples, m_total)
    assert (bitstrings_arr.sum(axis=1) == n_photons).all()

    # Exactly one photon per active pair/block
    active_part = bitstrings_arr[:, :m_active]
    blocks = active_part.reshape(n_samples, n_pairs_used, 2)
    assert (blocks.sum(axis=2) == 1).all()

    # Padding modes must be zero
    assert (bitstrings_arr[:, m_active:] == 0).all()


# Main script
if __name__ == "__main__":
    # Load rankings
    rankings = load_sushi3a_order(FILE_PATH, n_items=N_ITEMS)
    n_samples = rankings.shape[0]

    if N_TRAIN + N_TEST > n_samples:
        raise ValueError(
            f"N_TRAIN + N_TEST = {N_TRAIN + N_TEST}, but only "
            f"{n_samples} samples are available."
        )

    print("rankings shape:", rankings.shape)
    print("first ranking:", rankings[0])

    # Choose pairwise comparisons
    chosen_pairs = choose_pairs(
        n_items=N_ITEMS,
        n_pairs_used=N_PAIRS_USED,
        seed=SEED,
    )

    print("Chosen pairs:")
    for idx, pair in enumerate(chosen_pairs):
        print(idx, pair)

    # Encode rankings into pairwise block bitstrings
    bitstrings_arr = encode_rankings_pairwise_blocks(
        rankings=rankings,
        chosen_pairs=chosen_pairs,
        m_total=M_TOTAL,
    )

    print("bitstrings_arr shape:", bitstrings_arr.shape)
    print("first bitstring:", bitstrings_arr[0])
    print("Hamming weights:", np.unique(bitstrings_arr.sum(axis=1)))

    # Check dataset
    check_dataset(
        bitstrings_arr=bitstrings_arr,
        n_samples=n_samples,
        m_total=M_TOTAL,
        m_active=M_ACTIVE,
        n_pairs_used=N_PAIRS_USED,
        n_photons=N_PHOTONS,
    )

    print("All checks passed.")
    print("Final dataset lives in m =", M_TOTAL, "and n =", N_PHOTONS)

    # Shuffle and split train/test
    rng = np.random.default_rng(SEED)
    rng.shuffle(bitstrings_arr)

    X_train = bitstrings_arr[:N_TRAIN]
    X_test = bitstrings_arr[N_TRAIN:N_TRAIN + N_TEST]

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Train Hamming weights:", np.unique(X_train.sum(axis=1)))
    print("Test Hamming weights:", np.unique(X_test.sum(axis=1)))

    # Save
    os.makedirs(SAVE_PATH, exist_ok=True)

    train_file = os.path.join(
        SAVE_PATH,
        f"sushi_pairwise_block_train_m{M_TOTAL}_n{N_PHOTONS}.csv",
    )
    test_file = os.path.join(
        SAVE_PATH,
        f"sushi_pairwise_block_test_m{M_TOTAL}_n{N_PHOTONS}.csv",
    )
    pairs_file = os.path.join(
        SAVE_PATH,
        f"sushi_pairwise_chosen_pairs_seed{SEED}.csv",
    )

    np.savetxt(train_file, X_train, delimiter=",", fmt="%d")
    np.savetxt(test_file, X_test, delimiter=",", fmt="%d")
    np.savetxt(pairs_file, np.array(chosen_pairs, dtype=int), delimiter=",", fmt="%d")

    print("Saved files:")
    print(train_file)
    print(test_file)
    print(pairs_file)
