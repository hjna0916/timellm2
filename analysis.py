import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

csv_path = './dataset/MY_MD/FeN4_localcoords.csv'
df = pd.read_csv(csv_path)

T = len(df)
seq_len = 64
label_len = 48
pred_len = 12

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
result_path = './checkpoints/long_term_forecast_MD_local_TimeLLM_MY_MD_ftM_sl64_ll48_pl12_dm32_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-TimeLLM-Weather/results_prediction/'

# 2. pred.npy 와 true.npy 파일 로드
try:
    print(f"Loading results from {result_path}")
    preds = np.load(os.path.join(result_path, 'pred.npy'))
    trues = np.load(os.path.join(result_path, 'true.npy'))
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


# 4. 그래프 그리기 
# 32개 변수 중 보고 싶은 변수 인덱스
target_variable_index = 0 
# 보고 싶은 샘플 인덱스 
sample_index = 0 

print(f"Plotting Sample #{sample_index}, Variable #{target_variable_index}...")

plt.figure(figsize=(15, 6))
plt.plot(trues[sample_index, :, target_variable_index], label='Actual Coord (true.npy)', marker='o')
plt.plot(preds[sample_index, :, target_variable_index], label='Predicted Coord (pred.npy)', marker='x')
plt.title(f'Test Set - Sample {sample_index} - Variable {target_variable_index} (e.g., Fe_x)')
plt.xlabel('Future Timesteps')
plt.ylabel('Coordinate (Fe-centered)')
plt.legend()

# 5. 그래프를 이미지 파일로 저장
plot_filename = os.path.join(result_path, 'prediction_plot.png')
plt.savefig(plot_filename)
print(f"Plot saved to {plot_filename}")

plt.show()