import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------------
# 1. 读取数据（保持不变）
# ------------------------------
read_file = r''
results_df_path = r''
df = pd.read_csv(read_file,
                 parse_dates=['date'], index_col='date')
df = df.fillna(0)
print("数据集形状:", df.shape)

# ------------------------------
# 2. 构造 OT 的滞后特征（在全数据集上）
# ------------------------------
lags = [1, 2, 3, 6, 12, 24]          # 可根据需要调整滞后阶数
for lag in lags:
    df[f'OT_lag{lag}'] = df['OT'].shift(lag)
df.fillna(0, inplace=True)            # 填充 shift 产生的 NaN

# ------------------------------
# 3. 时间特征
# ------------------------------
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['quarter'] = df.index.quarter

# ------------------------------
# 4. 特征列与目标变量
# ------------------------------
original_features = ['HUFL', 'HULL', 'MULL', 'LUFL', 'LULL']
time_features = ['year', 'month', 'day', 'hour', 'dayofweek', 'quarter']
lag_features = [f'OT_lag{lag}' for lag in lags]
feature_cols = original_features + time_features + lag_features

X = df[feature_cols]
y = df['OT']

# ------------------------------
# 5. 按时间顺序划分 80% 训练，10% 验证，10% 测试
# ------------------------------
n = len(X)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]
X_val = X.iloc[train_end:val_end]
y_val = y.iloc[train_end:val_end]
X_test = X.iloc[val_end:]
y_test = y.iloc[val_end:]

print(f"训练集: {len(X_train)} ({len(X_train)/n:.1%})")
print(f"验证集: {len(X_val)} ({len(X_val)/n:.1%})")
print(f"测试集: {len(X_test)} ({len(X_test)/n:.1%})")

# ------------------------------
# 6. 定义模型参数模板（随机种子每次不同）
# ------------------------------
base_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    # 'random_state' 将在每次循环中动态赋值
}

# ------------------------------
# 7. 重复训练 3 次，收集预测结果
# ------------------------------
n_runs = 3
predictions = []          # 存储每次的预测值（列表，每个元素是 np.ndarray）

for run in range(n_runs):
    print(f"\n========== 第 {run+1} 次训练 ==========")
    
    # 每次使用不同的随机种子，确保结果具有随机性（防止偶然）
    current_params = base_params.copy()
    current_params['random_state'] = 42 + run   # 42, 43, 44
    
    # 创建 LightGBM 数据集
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    # 训练模型（早停基于验证集）
    model = lgb.train(
        current_params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=['train', 'valid'],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )
    
    # 在测试集上预测
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    predictions.append(y_pred)

# 转换为 numpy 数组，形状 (n_runs, len(y_test))
predictions = np.array(predictions)   # shape: (3, n_samples)

# ------------------------------
# 8. 计算平均预测值及最终评估指标
# ------------------------------
y_pred_mean = np.mean(predictions, axis=0)   # 按列平均，得到每个样本的平均预测值

mae = mean_absolute_error(y_test, y_pred_mean)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
r2 = r2_score(y_test, y_pred_mean)

print("\n" + "="*40)
print("=== 基于 3 次预测平均值的最终评估结果 ===")
print(f"MAE : {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2  : {r2:.4f}")

# ------------------------------
# 9. 保存预测结果（真实值 + 平均预测值 + 残差）
# ------------------------------
results_df = pd.DataFrame({
    'true_value': y_test,
    'predicted_value': y_pred_mean,
    'residual': y_test - y_pred_mean
})

# 可选：同时保存每次的预测值（便于分析方差）
for i in range(n_runs):
    results_df[f'pred_run_{i+1}'] = predictions[i]

results_df.to_csv(results_df_path,
                  index=False)

print(f"\n结果已保存至：{results_df_path}")
