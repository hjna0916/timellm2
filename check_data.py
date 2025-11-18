import numpy as np
import os

result_path = './checkpoints/long_term_forecast_MD_Fe-O_distances_TimeLLM_MY_MD_ftM_sl64_ll48_pl12_dm32_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-TimeLLM-Weather/results_prediction/'

# 2. NumPy를 사용해 .npy 파일 로드
try:
    # 'pred.npy' 파일을 불러와서 'preds' 변수에 저장
    preds = np.load(os.path.join(result_path, 'pred.npy'))

    # 'true.npy' 파일을 불러와서 'trues' 변수에 저장
    trues = np.load(os.path.join(result_path, 'true.npy'))

    # 3. 로드 성공 확인 (데이터의 '모양' 출력)
    print("파일 로드 성공!")
    print(f"예측값 (preds) Shape: {preds.shape}")
    print(f"실제값 (trues) Shape: {trues.shape}")

    print("\n--- [데이터 내용 확인] ---")
    # 1. preds의 첫 번째 샘플 (0번) 데이터 출력
    print("\n[예측값 (preds)의 첫 번째 샘플]:")
    print(preds[0]) 
    # 2. trues의 첫 번째 샘플 (0번) 데이터 출력
    print("\n[실제값 (trues)의 첫 번째 샘플]:")
    print(trues[0])
    print("---------------------------------")

except FileNotFoundError:
    print(f"ERROR: 파일 경로를 찾을 수 없습니다.")
    print(f"경로가 정확한지 다시 확인하세요: {result_path}")