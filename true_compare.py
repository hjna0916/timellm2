import numpy as np

# 1) 경로 설정 (네 폴더 이름에 맞게)
base_path = './checkpoints/long_term_forecast_MD_Fe-O_distances2_TimeLLM_MY_MD_ftM_sl64_ll48_pl12_dm32_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-TimeLLM-Weather/results_prediction/true.npy'
gnn_path  = './checkpoints/long_term_forecast_MD_GNNlite_TimeLLM_MY_MD_ftM_sl64_ll48_pl12_dm32_nh8_el2_dl1_df32_fc3_ebtimeF_test_0-TimeLLM-Weather/results_prediction/true.npy'

true_base = np.load(base_path)
true_gnn  = np.load(gnn_path)

print("baseline true shape:", true_base.shape)
print("gnn-lite true shape:", true_gnn.shape)

print("same shape? ", true_base.shape == true_gnn.shape)

# 값이 거의 같은지 (부동소수 오차 허용)
print("allclose?    ", np.allclose(true_base, true_gnn))

# 혹시 미세하게 다를 때 최대 차이도 확인
print("max abs diff:", np.max(np.abs(true_base - true_gnn)))
