import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # 3D plot용 

csv_path = './dataset/MY_MD/FeN4_localcoords.csv'
df = pd.read_csv(csv_path)

T = len(df)
seq_len = 64
label_len = 48
pred_len = 1

num_train = int(T * 0.7)
num_test = int(T * 0.2)
num_vali = T - num_train - num_test

border1_test = T - num_test - seq_len  # = len(df_raw) - num_test - seq_len

sample_index = 0  # 지금 그린 그래프의 sample_index
s_begin = sample_index
s_end = s_begin + seq_len

pred_local_idx = np.arange(s_end, s_end + pred_len)       # test 세그먼트 내 index
pred_global_idx = border1_test + pred_local_idx           # 전체 CSV 내 index

print("전역 index:", pred_global_idx)
print("해당하는 date:")
print(df['date'].iloc[pred_global_idx].to_list())

# 1. 'results_prediction' 폴더 경로
result_path = './checkpoints/long_term_forecast_MD_local_re_TimeLLM_MY_MD_ftM_sl64_ll48_pl1_dm32_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-TimeLLM-Weather/results_prediction/'

# 2. pred.npy 와 true.npy 파일 로드
try:
    print(f"Loading results from {result_path}")
    preds = np.load(os.path.join(result_path, 'pred.npy'))
    trues = np.load(os.path.join(result_path, 'true.npy'))

    B, L, C = preds.shape  # C = 24
    zeros_fe = np.zeros((B, L, 3))      # Fe_x,y,z = 0
    preds = np.concatenate([zeros_fe, preds], axis=-1)  # (B, 1, 27)
    trues = np.concatenate([zeros_fe, trues], axis=-1)  # (B, 1, 27)

except FileNotFoundError:
    print(f"ERROR: 파일 경로를 찾을 수 없습니다. '{result_path}' 경로가 정확한지 다시 확인하세요.")
    exit()

print(f"Prediction shape: {preds.shape}") 
print(f"True shape: {trues.shape}")    

# 3. MSE (평균 제곱 오차) 직접 계산하기
mse = np.mean((preds - trues)**2)
print("----------------------------------------------------")
print(f"계산된 MSE (Test Loss): {mse:.7f}")
print("----------------------------------------------------")

# ===== 좌표 구조 정보 정리 =====
num_samples, pred_len_from_file, num_vars = preds.shape
assert pred_len_from_file == pred_len, f"pred_len mismatch: {pred_len_from_file} vs {pred_len}"

assert num_vars % 3 == 0, "마지막 차원은 반드시 3의 배수여야 합니다 (x,y,z)"
num_atoms = num_vars // 3

# 9개 원자라고 가정 (Fe, N1~N4, O1~O4)
if num_atoms == 9:
    atom_names = [
        "Fe",
        "N1", "N2", "N3", "N4",
        "O1", "O2", "O3", "O4",
    ]
else:
    atom_names = [f"Atom{i+1}" for i in range(num_atoms)]
    print("경고: 예상과 다른 원자 개수입니다. atom_names를 자동으로 생성했습니다:", atom_names)

# ============================
# 4-1. 한 원자의 (x,y,z) 시간에 따른 변화 비교
# ============================

sample_index = 0      # 보고 싶은 샘플 인덱스
atom_index = 1        # 0: Fe, 1: N1, ..., 5: O1 등
atom_name = atom_names[atom_index]

# 해당 원자의 x,y,z만 추출: (pred_len, 3)
true_atom = trues[sample_index, :, atom_index*3:(atom_index+1)*3]  # (pred_len, 3)
pred_atom = preds[sample_index, :, atom_index*3:(atom_index+1)*3]  # (pred_len, 3)

timesteps = np.arange(pred_len)

plt.figure(figsize=(15, 5))
for dim, label in enumerate(['x', 'y', 'z']):
    plt.subplot(1, 3, dim+1)
    plt.plot(timesteps, true_atom[:, dim], marker='o', label='True')
    plt.plot(timesteps, pred_atom[:, dim], marker='x', label='Pred')
    plt.title(f'{atom_name} - {label}(t)')
    plt.xlabel('Future step')
    plt.ylabel(f'{label} (Å, Fe-centered)')
    plt.legend()

plt.suptitle(f'{atom_name} local coordinates over future steps (sample {sample_index})')
plt.tight_layout()

timeseries_plot_filename = os.path.join(
    result_path,
    f'coord_timeseries_{atom_name}_sample{sample_index}.png'
)
plt.savefig(timeseries_plot_filename)
print(f"Time-series plot saved to {timeseries_plot_filename}")
plt.show()

# ============================
# 4-2. 한 시점에서 9개 원자의 3D 구조 True vs Pred 비교
# ============================

sample_index = 0       # 같은 샘플로 볼지, 다른 샘플로 볼지 선택
time_index = -1        # 마지막 예측 시점 (-1) 또는 0 ~ pred_len-1 중 택1

# (num_vars,) → (num_atoms, 3)으로 reshape
true_step = trues[sample_index, time_index, :].reshape(num_atoms, 3)
pred_step = preds[sample_index, time_index, :].reshape(num_atoms, 3)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# True 구조 (동그라미)
ax.scatter(true_step[:, 0], true_step[:, 1], true_step[:, 2],
           label='True', marker='o')

# Pred 구조 (x표)
ax.scatter(pred_step[:, 0], pred_step[:, 1], pred_step[:, 2],
           label='Pred', marker='x')

# 각 원자 이름 라벨 (True 위치 기준으로)
for i, name in enumerate(atom_names):
    ax.text(true_step[i, 0], true_step[i, 1], true_step[i, 2],
            f"T-{name}", fontsize=8)
    ax.text(pred_step[i, 0], pred_step[i, 1], pred_step[i, 2],
            f"P-{name}", fontsize=8)

ax.set_title(f"Local 3D coords (sample {sample_index}, step {time_index})")
ax.set_xlabel("x (Å, Fe-centered)")
ax.set_ylabel("y (Å, Fe-centered)")
ax.set_zlabel("z (Å, Fe-centered)")
ax.legend()

plt.tight_layout()

struct_plot_filename = os.path.join(
    result_path,
    f'coord_3D_sample{sample_index}_t{time_index}.png'
)
plt.savefig(struct_plot_filename)
print(f"3D structure plot saved to {struct_plot_filename}")
plt.show()
