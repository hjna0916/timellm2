import numpy as np
import pandas as pd

df = pd.read_csv('./dataset/MY_MD/FeN4_localcoords.csv')
data = df.drop(columns=['date']).values   # (T, 27)

# test 구간 인덱스 (학습 때 쓰던 border1_test 그대로 사용)
T = len(df)
seq_len = 64
pred_len = 1
num_train = int(T * 0.7)
num_test = int(T * 0.2)
border1_test = T - num_test - seq_len

test_data = data[border1_test:]  # (test_len, 27)

# naive: y(t+1) = y(t)
y_true = test_data[1:, :]        # t+1
y_pred = test_data[:-1, :]       # t

mse_naive = np.mean((y_true - y_pred)**2)
print("Naive baseline MSE:", mse_naive)

