import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Need to prepare this file with pre-processing script
    preprocessed_data_path = "datasets/bioinformatics/lincs2020_trt_xpr_lm978_subset10k.tsv"

    # Choose dataset dimensions
    number_samples = 10000 
    frac_train = 0.7
    m = 978
    n = 25
    selection_method = "mad" # How to choose m among 978
    polarity = "up"

    df = pd.read_csv(preprocessed_data_path, sep="\t", index_col=0)
    df = df.sort_index()
    genes_total, available_samples = df.shape
    print(f"Loaded: {preprocessed_data_path}  shape={df.shape}  (genes x samples)")

    # Split into train and test sets before processing to avoid leakage
    dfT = df.T
    train_set = dfT.sample(frac=frac_train, random_state=42)
    test_set = dfT.drop(train_set.index)
    # Check shape
    train_set.shape, test_set.shape
    
    # Select m rows on train_set
    if m < genes_total:
        if selection_method == "mad":
            med = train_set.median(axis=0)
            metric = (train_set.sub(med, axis=1)).abs().median(axis=0)
            universe = metric.sort_values(ascending=False).head(m).index.tolist()
            universe.sort()
        elif selection_method == "random":
            rng = np.random.default_rng(seed = 42)
            universe = rng.choice(train_set.columns.values, size=m, replace=False).tolist()
            universe.sort()
        else:
            raise ValueError("selection_method must be 'mad' or 'random'")
        train_set = train_set[universe]
        test_set = test_set[universe]
        print(f"Selected universe of size m={m} using {selection_method}")
    elif m == genes_total:
        print(f"Using all rows as universe (m={m})")
    else:
        raise ValueError(f"m={m} is larger than number of rows in file ({genes_total})")

    # Choose score view based on polarity
    if polarity == "up":
        scores_train = train_set.values.astype(float).copy()
        scores_test = test_set.values.astype(float).copy()
    elif polarity == "down":
        scores_train = -train_set.values.astype(float).copy()
        scores_test = -test_set.values.astype(float).copy()
    elif polarity == "abs":
        scores_train = np.abs(train_set.values.astype(float).copy())
        scores_test = np.abs(test_set.values.astype(float).copy())
    else:
        raise ValueError("polarity must be 'up', 'down', or 'abs'")

    # Get top-n per row 
    topn_train = np.argpartition(-scores_train, n-1, axis=1)[:, :n] 
    topn_test = np.argpartition(-scores_test, n-1, axis=1)[:, :n] 

    # Create output bitstring array
    n_samples_train = int(number_samples*frac_train)
    X_train = np.zeros((n_samples_train, m), dtype=np.uint8)
    # Set ones
    for sample_index in range(n_samples_train):
        colscores = scores_train[sample_index, :]
        top_n = topn_train[sample_index, :]
        X_train[sample_index, top_n] = 1

    
    n_samples_test = int(number_samples*(1-frac_train))
    X_test = np.zeros((n_samples_test, m), dtype=np.uint8)
    # Set ones
    for sample_index in range(n_samples_test):
        colscores = scores_test[sample_index, :]
        top_n = topn_test[sample_index, :]
        X_test[sample_index, top_n] = 1


    # Check there are n 1s in all bitstrings
    if np.any(X_train.sum(axis=1) != n) or np.any(X_test.sum(axis=1) != n):
        print('Some samples do not have the right Hamming weight!')
    
    # save
    path = './bioinformatics/'
    np.savetxt(path + 'bioinformatics_train_m' + str(m) + '_n' + str(n) + '.csv', X_train, delimiter=',',  fmt='%d')
    np.savetxt(path + 'bioinformatics_test_m' + str(m) + '_n' + str(n) + '.csv', X_train, delimiter=',',  fmt='%d')
    
