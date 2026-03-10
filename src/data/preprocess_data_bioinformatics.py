from cmapPy.pandasGEXpress.parse import parse
import pandas as pd
import numpy as np

if __name__ == "__main__":

    # Paths
    gctx_path = "datasets/bioinformatics/level5_beta_trt_xpr_n142901x12328.gctx"
    metadata_path = "datasets/bioinformatics/geneinfo_beta.txt"
    lm978_path = "datasets/bioinformatics/l1000_landmarks_978.csv"
    output_path = "datasets/bioinformatics/lincs2020_trt_xpr_lm978_subset5000.tsv"
    
    # Read the GCTX (downloaded from CLUE)
    gct = parse(gctx_path, convert_neg_666=True)
    df = gct.data_df
    # shape should be (12328, 142901)
    
    # Gene info metadata (downloaded from CLUE)
    metadata_df = pd.read_csv(metadata_path, sep="\t")
    
    # Check
    metadata_df['feature_space'].value_counts()
    
    # Keep landmark gene codes and save
    lm = metadata_df.loc[metadata_df["feature_space"] == "landmark", ["gene_symbol", "gene_id"]]
    lm = lm.drop_duplicates()
    lm.to_csv(lm978_path, index=False, header=True)
    print(f"Saved {len(lm)} landmark genes")
    
    # Filter the 978 landmark genes from GCTX df
    lm_genes_list = pd.read_csv(lm978_path)["gene_id"].tolist()
    lm_genes_list = [str(gene_id) for gene_id in lm_genes_list]
    df_lm = df.loc[df.index.intersection(lm_genes_list)]
    
    # Randomly sample 10000 signatures
    rng = np.random.default_rng(42)
    subset_cols = rng.choice(df_lm.columns, size=10000, replace=False)
    subset = df_lm[subset_cols]
    
    # Save
    subset.to_csv(output_path, sep="\t")
