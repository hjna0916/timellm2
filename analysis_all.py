import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 'results_prediction' 폴더 경로
result_path = './checkpoints/long_term_forecast_MD_Fe-O_distances_TimeLLM_MY_MD_ftM_sl64_ll48_pl12_dm32_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-TimeLLM-Weather/results_prediction/'
 

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
     
# 3. MSE 계산
mse = np.mean((preds - trues)**2)
print("----------------------------------------------------")
print(f"계산된 MSE (Test Loss): {mse:.7f}")
print(f"(참고) 10.7시간 학습시 출력된 Test Loss: 0.0367829")
print("----------------------------------------------------")


# 4. 32개 변수 그래프 한 번에 그리기

num_variables = 32 #
sample_index = 0  # 0번째 샘플 데이터로 비교

# 8x4 격자, 전체 이미지 크기 설정
fig, axes = plt.subplots(8, 4, figsize=(20, 40)) 
fig.suptitle(f'All 32 Variables (Sample {sample_index}): Actual vs. Predicted', fontsize=24, y=1.02)

variable_index = 0
for i in range(8): # 8줄
    for j in range(4): # 4칸
        if variable_index < num_variables:
            # (i, j) 위치의 작은 그래프 선택
            ax = axes[i, j]

            # 'Actual' (true.npy) 값 그리기
            ax.plot(trues[sample_index, :, variable_index], label='Actual (True)', marker='o', markersize=4)
            # 'Predicted' (pred.npy) 값 그리기
            ax.plot(preds[sample_index, :, variable_index], label='Predicted (Pred)', marker='x', markersize=4)

            ax.set_title(f'Variable #{variable_index}')
            ax.legend(fontsize='small')

            variable_index += 1
        else:
            ax.axis('off') # 32개가 넘어가면 빈칸으로 둠

plt.tight_layout() # 그래프들이 겹치지 않게 정렬

# 5. 그래프를 '하나의 큰' 이미지 파일로 저장
plot_filename = os.path.join(result_path, 'all_variables_plot.png')
plt.savefig(plot_filename)
print(f"Plot saved to {plot_filename}")

plt.show()