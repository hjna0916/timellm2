import pandas as pd


# concat_dist
df_n = pd.read_csv("Fe_all_N_distances.csv")
df_o = pd.read_csv("Fe_all_O_distances.csv")  

df_dist = pd.concat([df_n, df_o], axis=1)    

df_dist.to_csv("md_distance_only.csv", index=False)
print(df_dist.head())

# concat dist + features
df_dist = pd.read_csv("md_distance_only.csv")                    
df_feat = pd.read_csv("Fe_gnnlite_interaction_features.csv")     

if "Step" in df_feat.columns:
    df_feat = df_feat.drop(columns=["Step"])

df_combined = pd.concat([df_dist, df_feat], axis=1)

df_combined.to_csv("md_data_gnn_lite_combined.csv", index=False)
print(df_combined.shape)
print(df_combined.head())