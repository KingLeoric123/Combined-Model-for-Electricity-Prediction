read_file = r"C:\Users\Terry\Desktop\ABC\立项\2026-交通+教学\2026-统计建模大赛\数据集\原始数据集\ETTm1.csv"
results_df_path = r"C:\Users\Terry\Desktop\ABC\立项\2026-交通+教学\2026-统计建模大赛\数据集\用于绘图的数据集\LSTM\add.csv"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# -------------------- 1. 设置设备 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# -------------------- 2. 数据加载与特征工程（只执行一次） --------------------
df = pd.read_csv(read_file,
                 parse_dates=['date'], index_col='date')
df = df.fillna(0)
print("数据集形状:", df.shape)

# 添加时间特征
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month

feature_cols = ['HUFL', 'HULL', 'MULL', 'LUFL', 'LULL', 'hour', 'dayofweek', 'month']
target_col = 'OT'
look_back = 24

# 构造滑动窗口序列
def create_sequences(data, target, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data.iloc[i:i+look_back].values)
        y.append(data[target].iloc[i+look_back])
    return np.array(X), np.array(y)

input_features = feature_cols + [target_col]
data_for_seq = df[input_features].copy()

# 标准化（特征+目标一起缩放，便于后续统一反标准化目标）
scaler_all = MinMaxScaler()
scaled_data = scaler_all.fit_transform(data_for_seq)

X, y = create_sequences(pd.DataFrame(scaled_data, columns=input_features), target_col, look_back)

# 划分训练集和测试集（80% / 20%）
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"训练样本数: {X_train.shape[0]}, 测试样本数: {X_test.shape[0]}")
print(f"输入形状: {X_train.shape[1]} 时间步 × {X_train.shape[2]} 特征")

# 转换为 PyTorch 张量（保持不变，所有运行共用）
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# 从训练集中划分验证集（后10%）
val_ratio = 0.1
val_size = int(len(X_train_t) * val_ratio)
train_size = len(X_train_t) - val_size
X_train_sub = X_train_t[:train_size]
y_train_sub = y_train_t[:train_size]
X_val = X_train_t[train_size:]
y_val = y_train_t[train_size:]

# 创建 DataLoader（不 shuffle，保持时序）
batch_size = 32
train_dataset = TensorDataset(X_train_sub, y_train_sub)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 目标变量的缩放器（用于反标准化）
scaler_y = MinMaxScaler()
scaler_y.fit(df[[target_col]])

# -------------------- 3. 定义 LSTM 模型 --------------------
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        return self.fc(last_out)

# -------------------- 4. 单次实验函数（完整训练 + 最佳模型回溯） --------------------
def run_experiment(run_id, seed=None):
    """执行一次完整的训练和评估，返回反标准化后的预测值及指标"""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    # 重新初始化模型和优化器
    input_size = X_train.shape[2]
    model = LSTMPredictor(input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 30
    train_losses = []
    val_losses = []
    best_val_loss = np.inf
    #best_model_path = f'best_lstm_model_run{run_id}.pth'
    
    print(f"\n开始训练 Run {run_id} ...")
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * X_batch.size(0)
        epoch_train_loss /= train_size
        train_losses.append(epoch_train_loss)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                epoch_val_loss += loss.item() * X_batch.size(0)
        epoch_val_loss /= val_size
        val_losses.append(epoch_val_loss)
        
        # 保存验证损失最小的模型（回溯到最佳轮次）
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), best_model_path)
            # 可选打印
            # print(f'  Epoch {epoch+1}: Validation loss improved to {epoch_val_loss:.6f}. Model saved.')
        
        if (epoch+1) % 10 == 0:
            print(f'Run {run_id} Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}')
    
    # 加载最佳模型权重（回溯）
    model.load_state_dict(torch.load(best_model_path))
    print(f"Run {run_id} 完成，最佳验证损失为 {best_val_loss:.6f}，已加载该模型权重。")
    # 可选：删除保存的模型文件
    # os.remove(best_model_path)
    
    # 测试集预测
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_test_t).cpu().numpy().flatten()
    
    # 反标准化
    y_test_np = y_test_t.cpu().numpy().flatten()
    y_test_inv = scaler_y.inverse_transform(y_test_np.reshape(-1, 1)).flatten()
    y_pred_inv = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    
    # 计算指标
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    print(f"Run {run_id} 结果: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
    
    return y_pred_inv, mae, rmse, r2

# -------------------- 5. 运行三次实验并取平均 --------------------
n_runs = 3
all_preds = []
all_mae = []
all_rmse = []
all_r2 = []

for i in range(1, n_runs+1):
    seed = 42 + i   # 不同种子保证不同的随机初始化
    y_pred, mae, rmse, r2 = run_experiment(run_id=i, seed=seed)
    all_preds.append(y_pred)
    all_mae.append(mae)
    all_rmse.append(rmse)
    all_r2.append(r2)

# 计算平均预测值和平均指标
avg_pred = np.mean(all_preds, axis=0)
avg_mae = np.mean(all_mae)
avg_rmse = np.mean(all_rmse)
avg_r2 = np.mean(all_r2)

print("\n" + "="*40)
print("=== 三次运行平均结果 ===")
print(f"Average MAE : {avg_mae:.4f}")
print(f"Average RMSE: {avg_rmse:.4f}")
print(f"Average R2  : {avg_r2:.4f}")

# 真实值（三次运行相同，取第一次即可）
true_vals = scaler_y.inverse_transform(y_test_t.cpu().numpy().flatten().reshape(-1, 1)).flatten()

# 保存平均预测结果到 CSV
results_df = pd.DataFrame({
    'true_value': true_vals,
    'predicted_value': avg_pred,
    'residual': true_vals - avg_pred
})
save_path = results_df_path
results_df.to_csv(save_path, index=False)
print(f"\n平均预测结果已保存至: {save_path}")
