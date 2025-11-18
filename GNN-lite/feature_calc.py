import ase.io
import pandas as pd
import numpy as np

# --- 파일 이름 지정 ---
POSCAR_FILE = 'POSCAR'
XDATCAR_FILE = 'XDATCAR'

print("Reading files...")

try:
    images = ase.io.read(XDATCAR_FILE, index=':')
    initial_structure = ase.io.read(POSCAR_FILE)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

print("Files read successfully.")

# --- 원자 인덱스 정의 ---
# POSCAR 파일 정보: C=94, N=4, Fe=1, H=64, O=32
fe_index = 98                        # Fe
n_indices = list(range(94, 98))      # N 4개
o_indices = list(range(163, 195))    # O 32개 (163~194)

# GNN-lite 피처를 저장할 리스트
feature_rows = []

# 하이퍼파라미터(컷오프 등) 정의
cutoff_fe_n = 2.1    # Fe-N CN cutoff (Å)
cutoff_fe_o = 2.5    # Fe-O CN cutoff (Å)
shell1 = 3.0         # water-like O shell1
shell2 = 4.0
shell3 = 5.0

print("Calculating GNN-lite interaction features...")

for i, atoms in enumerate(images):
    step = i + 1
    pos = atoms.get_positions()

    # --------- Fe-N 거리들 ---------
    d_fe_n = []
    for n_idx in n_indices:
        d = atoms.get_distance(fe_index, n_idx, mic=True)
        d_fe_n.append(d)
    d_fe_n = np.array(d_fe_n)  # shape: (4,)

    cn_fe_n = np.sum(d_fe_n < cutoff_fe_n)
    fe_n_mean = d_fe_n.mean()
    fe_n_min = d_fe_n.min()
    fe_n_std = d_fe_n.std()

    # --------- Fe-O 거리들 ---------
    d_fe_o = []
    for o_idx in o_indices:
        d = atoms.get_distance(fe_index, o_idx, mic=True)
        d_fe_o.append(d)
    d_fe_o = np.array(d_fe_o)  # shape: (32,)

    cn_fe_o = np.sum(d_fe_o < cutoff_fe_o)
    fe_o_mean = d_fe_o.mean()
    fe_o_min = d_fe_o.min()
    fe_o_std = d_fe_o.std()

    # --------- shell 별 O 개수 (3,4,5 Å) ---------
    shell1_mask = d_fe_o < shell1
    shell2_mask = (d_fe_o >= shell1) & (d_fe_o < shell2)
    shell3_mask = (d_fe_o >= shell2) & (d_fe_o < shell3)

    shell1_count = np.sum(shell1_mask)
    shell2_count = np.sum(shell2_mask)
    shell3_count = np.sum(shell3_mask)

    # shell1 내 Fe-O 통계
    if shell1_count > 0:
        shell1_mean = d_fe_o[shell1_mask].mean()
        shell1_min = d_fe_o[shell1_mask].min()
        shell1_std = d_fe_o[shell1_mask].std()
    else:
        shell1_mean = 0.0
        shell1_min = 0.0
        shell1_std = 0.0

    # --------- Fe out-of-plane (N4 평면에서 얼마나 튀어나왔는지) ---------
    n_pos = pos[n_indices]      # (4,3)
    fe_pos = pos[fe_index]      # (3,)

    # 간단히 N0, N1, N2로 평면 정의 (N들이 거의 평면이라고 가정)
    v1 = n_pos[1] - n_pos[0]
    v2 = n_pos[2] - n_pos[0]
    normal = np.cross(v1, v2)
    norm_normal = np.linalg.norm(normal)

    if norm_normal > 0:
        normal = normal / norm_normal
        # 점-평면 거리: |n · (Fe - N0)|
        fe_out_of_plane = abs(np.dot(normal, fe_pos - n_pos[0]))
    else:
        fe_out_of_plane = 0.0

    # --------- 한 스텝의 피처 묶기 ---------
    row = {
        "Step": step,
        # Fe-N
        "cn_fe_n": cn_fe_n,
        "fe_n_mean": fe_n_mean,
        "fe_n_min": fe_n_min,
        "fe_n_std": fe_n_std,
        # Fe-O
        "cn_fe_o": cn_fe_o,
        "fe_o_mean": fe_o_mean,
        "fe_o_min": fe_o_min,
        "fe_o_std": fe_o_std,
        # shells
        "shell1_count": shell1_count,
        "shell2_count": shell2_count,
        "shell3_count": shell3_count,
        "shell1_mean": shell1_mean,
        "shell1_min": shell1_min,
        "shell1_std": shell1_std,
        # geometry
        "fe_out_of_plane": fe_out_of_plane,
    }

    feature_rows.append(row)

print("Feature calculation completed.")

# DataFrame으로 저장
df_feat = pd.DataFrame(feature_rows)
output_filename = "Fe_gnnlite_interaction_features.csv"
df_feat.to_csv(output_filename, index=False)
print(f"-> Saved '{output_filename}' successfully.")

print("\n### Feature preview ###")
print(df_feat.head())
