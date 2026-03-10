import numpy as np

if __name__ == "__main__":

    # Download and save data from:
    # https://github.com/PrefLib/PrefLib-Data/blob/main/datasets/00014%20-%20sushi/00014-00000002.soi
    file_path = "datasets/preference_ranking/sushi.txt"
    
    with open(file_path, "r") as f:
        data = []
        for line in f:
            # Remove the "1:" prefix and any surrounding whitespace
            line = line.strip().replace("1:", "", 1).strip()
            # Split the line by commas and convert to integers
            numbers = list(map(int, line.split(",")))
            data.append(numbers)
    
    
    arr = np.array(data)
    
    n_samples = 5000
    length_bitstrings = 100
    bitstrings_arr = np.zeros((n_samples, length_bitstrings), dtype=int)
    
    rows = np.arange(n_samples)[:, None]
    bitstrings_arr[rows, arr - 1] = 1
    
    # Check there are 10 1s in all bitstrings
    if np.any(bitstrings_arr.sum(axis=1) != 10):
        print('Some samples do not have the right Hamming weight!')
    
    # Shuffle dataset
    rng = np.random.default_rng(seed=42)
    rng.shuffle(bitstrings_arr)
    
    # Split in train /test
    X_train = bitstrings_arr[:3000]
    X_test = bitstrings_arr[3000:]
    
    # save
    path = './preference_ranking/'
    np.savetxt(path + 'sushi_train.csv', X_train, delimiter=',',  fmt='%d')
    np.savetxt(path + 'sushi_test.csv', X_test, delimiter=',', fmt='%d')
