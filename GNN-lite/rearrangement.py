import pandas as pd

df = pd.read_csv("md_data_gnn_lite_combined.csv")

# date 떼고 나머지
date = df["date"]
feat = df.drop(columns=["date"])

# 지금 feat.columns 순서를 보고,
# Fe-O, Fe-N, feature 컬럼 이름들을 직접 리스트로 지정
fe_n_cols = ['Fe-N94_dist', 'Fe-N95_dist', 'Fe-N96_dist', 'Fe-N97_dist']   # Fe-N 4개 이름
fe_o_cols = ['Fe-O163_dist', 'Fe-O164_dist', 'Fe-O165_dist', 'Fe-O166_dist',
       'Fe-O167_dist', 'Fe-O168_dist', 'Fe-O169_dist', 'Fe-O170_dist',
       'Fe-O171_dist', 'Fe-O172_dist', 'Fe-O173_dist', 'Fe-O174_dist',
       'Fe-O175_dist', 'Fe-O176_dist', 'Fe-O177_dist', 'Fe-O178_dist',
       'Fe-O179_dist', 'Fe-O180_dist', 'Fe-O181_dist', 'Fe-O182_dist',
       'Fe-O183_dist', 'Fe-O184_dist', 'Fe-O185_dist', 'Fe-O186_dist',
       'Fe-O187_dist', 'Fe-O188_dist', 'Fe-O189_dist', 'Fe-O190_dist',
       'Fe-O191_dist', 'Fe-O192_dist', 'Fe-O193_dist', 'Fe-O194_dist',]   # Fe-O 32개 이름
feat_cols  = ['cn_fe_n', 'fe_n_mean', 'fe_n_min', 'fe_n_std', 'cn_fe_o', 'fe_o_mean',
       'fe_o_min', 'fe_o_std', 'shell1_count', 'shell2_count', 'shell3_count',
       'shell1_mean', 'shell1_min', 'shell1_std', 'fe_out_of_plane']  # 요약 피처 15개 이름

# 원하는 순서로 재배치: Fe-O -> Fe-N -> feature
feat_reordered = feat[fe_o_cols + fe_n_cols + feat_cols]

df_new = pd.concat([date, feat_reordered], axis=1)
df_new.to_csv("md_data_gnn_lite_with_date_reordered.csv", index=False)
