import kagglehub
import pandas as pd
import numpy as np

# Donwload raw data from Kaggle and save at location
#path = kagglehub.dataset_download("evanschreiner/netflix-movie-ratings")
#print("Path to dataset files:", path)

if __name__ == "__main__":

    # Load dataset
    df = pd.read_csv(path)
    
    # Filter
    df_dated = df[df['Date'] >= '2005-01-01'].copy()
    
    # Choose m and n
    m = 150
    n = 10
    
    # Take m movies that have been rated most often
    top_m_movies = (
        df['MovieId']
        #df_dated['MovieId']
        .value_counts()
        .head(m)
        .index
    )
    
    # Keep only those
    df_top = df_dated[df_dated['MovieId'].isin(top_m_movies)].copy()
    
    # Sort and take top n per user
    df_topn = (
        df_top
        .sort_values(['CustId', 'Rating', 'Date'], ascending=[True, False, False])
        .groupby('CustId')
        .head(n)
    )
    
    # Keep those that have indeed 10 favourite films and not less
    series_check = df_topn.groupby('CustId')['MovieId'].nunique()
    cust_n_counts = series_check[series_check.isin([n])].index
    df_topn_filtered = df_topn[df_topn['CustId'].isin(cust_n_counts)].copy()
    
    # Send to right format
    df_topn_simplified = df_topn_filtered[['CustId', 'MovieId']] .copy()
    
    onehot = (
        df_topn_simplified
        .assign(value=1)
        .pivot_table(index="CustId", columns="MovieId", values="value", fill_value=0)
    )
    
    np_onehot = onehot.to_numpy()
    
    if np.any(np_onehot.sum(axis=1) != n):
        print('Some samples do not have the right Hamming weight!')
    
    # Shuffle dataset
    rng = np.random.default_rng(seed=42)
    rng.shuffle(bitstrings_arr)
    
    # Split in train /test
    X_train = bitstrings_arr[:10000]
    X_test = bitstrings_arr[10000:20000]
    
    # save
    path = './preference_ranking/'
    np.savetxt(path + 'movie_m_' + str(m) + '_n_' + str(n) + '_train.csv', X_train, delimiter=',',  fmt='%d')
    np.savetxt(path + 'movie_m_' + str(m) + '_n_' + str(n) + '_test.csv', X_test, delimiter=',', fmt='%d')
