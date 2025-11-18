import ase.io
import pandas as pd
import numpy as np

# 분석할 파일 이름 지정
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
fe_index = 98                   # Fe 원자의 인덱스 (0부터 시작)
n_indices = list(range(94, 98))   # N 원자들의 인덱스
o_indices = list(range(163, 195)) # O 원자들의 인덱스

# 결과를 각각 저장할 리스트
fe_n_data = []
fe_o_data = []

print("Calculating distances...")

# 모든 프레임 순회
for i, atoms in enumerate(images):
    step = i + 1
    
    # 현재 스텝의 데이터를 저장할 딕셔너리
    n_dist_current_step = {'Step': step}
    o_dist_current_step = {'Step': step}
    
    # 1. 각 질소(N) 원자와의 거리를 계산
    for n_idx in n_indices:
        distance = atoms.get_distance(fe_index, n_idx, mic=True)
        column_name = f'Fe-{initial_structure[n_idx].symbol}{n_idx}_dist'
        n_dist_current_step[column_name] = distance
    fe_n_data.append(n_dist_current_step)
    
    # 2. 각 산소(O) 원자와의 거리를 계산
    for o_idx in o_indices:
        distance = atoms.get_distance(fe_index, o_idx, mic=True)
        column_name = f'Fe-{initial_structure[o_idx].symbol}{o_idx}_dist'
        o_dist_current_step[column_name] = distance
    fe_o_data.append(o_dist_current_step)

print("Distance calculations completed.")

# --- Fe-N 데이터셋 저장 ---
df_n = pd.DataFrame(fe_n_data)
output_filename_n = 'Fe_all_N_distances.csv'
df_n.to_csv(output_filename_n, index=False)
print(f"-> Saved '{output_filename_n}' successfully.")

# --- Fe-O 데이터셋 저장 ---
df_o = pd.DataFrame(fe_o_data)
output_filename_o = 'Fe_all_O_distances.csv'
df_o.to_csv(output_filename_o, index=False)
print(f"-> Saved '{output_filename_o}' successfully.")

print("\n### Fe-N 데이터셋 미리보기 ###")
print(df_n.head())
print("\n### Fe-O 데이터셋 미리보기 ###")
print(df_o.head())