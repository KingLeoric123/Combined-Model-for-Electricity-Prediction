read_file = r"C:\Users\Terry\Desktop\ABC\立项\2026-交通+教学\2026-统计建模大赛\数据集\用于测试模型稳健性的数据集\去除了椒盐噪声的数据集\cnoise_m1.csv"
results_df_path = None

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ====================== 1. 数据准备 ======================
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(data, target_col, feature_cols, seq_len, pred_len=1):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[feature_cols].iloc[i:i+seq_len].values)
        y.append(data[target_col].iloc[i+seq_len:i+seq_len+pred_len].values[-1])
    return np.array(X), np.array(y)

# ====================== 2. Transformer模型 ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerTemperaturePredictor(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1):
        super(TransformerTemperaturePredictor, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                                    dim_feedforward=dim_feedforward, 
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)   # 用于MC Dropout时额外激活
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        out = self.output_proj(x)
        return out

# ====================== 3. Monte-Carlo Dropout预测函数 ======================
def mc_dropout_predict(model, dataloader, n_iter=50, device='cuda'):
    """
    使用Monte Carlo Dropout进行不确定性估计
    返回：预测均值 (N,), 预测标准差 (N,)
    """
    model.train()   # 启用Dropout层
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            batch_preds = []
            for _ in range(n_iter):
                pred = model(X_batch).cpu().numpy()   # (batch, 1)
                batch_preds.append(pred)
            batch_preds = np.stack(batch_preds, axis=0)   # (n_iter, batch, 1)
            mean_pred = np.mean(batch_preds, axis=0)      # (batch, 1)
            std_pred = np.std(batch_preds, axis=0)        # (batch, 1)
            all_preds.append(np.concatenate([mean_pred, std_pred], axis=1))  # (batch, 2)
    all_preds = np.concatenate(all_preds, axis=0)   # (N, 2)
    return all_preds[:, 0], all_preds[:, 1]   # mean, std

# ====================== 4. 训练函数 ======================
def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, device='cuda', patience=10):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f'Epoch {epoch+1:3d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    model.load_state_dict(torch.load('best_model.pth'))
    return model, train_losses, val_losses

# ====================== 5. 主程序（运行三遍取平均，每遍使用MC Dropout） ======================
if __name__ == '__main__':
    n_runs = 10                    # 运行次数
    all_preds_means = []           # 存储每次运行的预测均值（已逆标准化）
    all_preds_stds = []            # 存储每次运行的预测标准差（原始尺度）
    all_trues = None               # 真实值（每次相同）
    run_metrics = []               # 存储每次运行的指标
    
    for run in range(n_runs):
        print(f'\n========== Run {run+1}/{n_runs} ==========')
        
        # -------------------- 加载数据 --------------------
        df = pd.read_csv(read_file, parse_dates=['date'])
        
        base_features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
        target_col = 'OT'
        feature_cols = base_features + [target_col]   # 包含历史OT
        
        print(f"使用的输入特征列：{feature_cols}")
        print(f"预测目标列：{target_col}")
        
        # 去除缺失值
        df = df.dropna(subset=feature_cols + [target_col])
        df = df.sort_values('date')
        
        # -------------------- 划分训练/验证/测试集（按时间顺序） --------------------
        train_ratio = 0.7
        val_ratio = 0.1
        test_ratio = 0.2
        n_total = len(df)
        train_end = int(train_ratio * n_total)
        val_end = train_end + int(val_ratio * n_total)
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        print(f"训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}, 测试集大小: {len(test_df)}")
        
        # -------------------- 数据标准化 --------------------
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        train_X_raw = train_df[feature_cols].values
        train_y_raw = train_df[target_col].values.reshape(-1, 1)
        scaler_X.fit(train_X_raw)
        scaler_y.fit(train_y_raw)
        
        train_df_scaled = train_df.copy()
        val_df_scaled = val_df.copy()
        test_df_scaled = test_df.copy()
        train_df_scaled[feature_cols] = scaler_X.transform(train_df[feature_cols])
        val_df_scaled[feature_cols] = scaler_X.transform(val_df[feature_cols])
        test_df_scaled[feature_cols] = scaler_X.transform(test_df[feature_cols])
        train_df_scaled[target_col] = scaler_y.transform(train_df[[target_col]]).flatten()
        val_df_scaled[target_col] = scaler_y.transform(val_df[[target_col]]).flatten()
        test_df_scaled[target_col] = scaler_y.transform(test_df[[target_col]]).flatten()
        
        # -------------------- 创建滑动窗口序列 --------------------
        seq_len = 24
        pred_len = 1
        X_train, y_train = create_sequences(train_df_scaled, target_col, feature_cols, seq_len, pred_len)
        X_val, y_val = create_sequences(val_df_scaled, target_col, feature_cols, seq_len, pred_len)
        X_test, y_test = create_sequences(test_df_scaled, target_col, feature_cols, seq_len, pred_len)
        
        print(f"训练样本数: {X_train.shape}, 验证样本数: {X_val.shape}, 测试样本数: {X_test.shape}")
        
        # -------------------- 创建 DataLoader --------------------
        batch_size = 64
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # -------------------- 模型训练 --------------------
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        input_dim = len(feature_cols)
        model = TransformerTemperaturePredictor(
            input_dim=input_dim,
            d_model=64,
            nhead=4,
            num_layers=3,
            dim_feedforward=128,
            dropout=0.1
        )
        print(model)
        
        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, epochs=100, lr=1e-3, device=device, patience=10
        )
        
        # -------------------- 测试评估（使用Monte Carlo Dropout） --------------------
        # 获取预测均值（标准化尺度）和标准差（标准化尺度）
        pred_mean_scaled, pred_std_scaled = mc_dropout_predict(model, test_loader, n_iter=50, device=device)
        # 真实值（标准化尺度）
        y_true_scaled = np.concatenate([y_batch.numpy() for _, y_batch in test_loader], axis=0).flatten()
        
        # 逆标准化到原始尺度
        y_true = scaler_y.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
        pred_mean = scaler_y.inverse_transform(pred_mean_scaled.reshape(-1, 1)).flatten()
        # 标准差需要缩放（因为标准化是线性变换：y_orig = y_scaled * std + mean，所以标准差也乘以 std）
        pred_std = pred_std_scaled * scaler_y.scale_[0]
        
        # 计算性能指标
        rmse = np.sqrt(np.mean((pred_mean - y_true)**2))
        mae = np.mean(np.abs(pred_mean - y_true))
        ss_res = np.sum((y_true - pred_mean)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0
        
        # 平均预测标准差（衡量整体不确定性）
        avg_std = np.mean(pred_std)
        print(f"测试集性能（MC Dropout均值预测）: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        print(f"平均预测标准差: {avg_std:.4f} °C")
        
        # 保存本次运行结果
        run_metrics.append({'rmse': rmse, 'mae': mae, 'r2': r2, 'avg_std': avg_std})
        if all_trues is None:
            all_trues = y_true
        all_preds_means.append(pred_mean)
        all_preds_stds.append(pred_std)
        
        # 可选：预测下一时刻油温（使用最后一次模型，确定性预测）
        if run == n_runs - 1:
            last_seq = test_df[feature_cols].iloc[-seq_len:].values
            def predict_future(model, last_sequence, scaler_X, scaler_y, device='cuda'):
                model.eval()   # 确定性预测，关闭dropout
                last_seq_scaled = scaler_X.transform(last_sequence)
                input_tensor = torch.tensor(last_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_scaled = model(input_tensor).cpu().numpy()
                pred = scaler_y.inverse_transform(pred_scaled)
                return pred[0,0]
            next_temp = predict_future(model, last_seq, scaler_X, scaler_y, device)
            print(f"基于最后{seq_len}个历史数据，确定性预测下一时刻油温: {next_temp:.2f} °C")
    
    preds_avg = np.mean(all_preds_means, axis=0)          # 三次运行预测均值的平均
    preds_std_avg = np.mean(all_preds_stds, axis=0)       # 三次运行标准差的平均（反映模型自身不确定性）
    # 也可以计算三次运行预测值之间的标准差（反映模型间不确定性）
    preds_inter_std = np.std(all_preds_means, axis=0)
    
    # 最终性能（使用集成预测均值）
    rmse_final = np.sqrt(np.mean((preds_avg - all_trues)**2))
    mae_final = np.mean(np.abs(preds_avg - all_trues))
    ss_res = np.sum((all_trues - preds_avg)**2)
    ss_tot = np.sum((all_trues - np.mean(all_trues))**2)
    r2_final = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0
    
    # 打印各次运行的指标统计
    print("\n========== 各次运行性能（MC Dropout） ==========")
    for i, m in enumerate(run_metrics):
        print(f"Run {i+1}: RMSE={m['rmse']:.4f}, MAE={m['mae']:.4f}, R2={m['r2']:.4f}, AvgStd={m['avg_std']:.4f}")
    print(f"平均指标: RMSE={np.mean([m['rmse'] for m in run_metrics]):.4f} ± {np.std([m['rmse'] for m in run_metrics]):.4f}")
    
    print("\n========== 最终集成结果（三次预测平均） ==========")
    print(f"RMSE (集成预测): {rmse_final:.4f}")
    print(f"MAE  (集成预测): {mae_final:.4f}")
    print(f"R2   (集成预测): {r2_final:.4f}")
    print(f"平均MC Dropout标准差（反映单次模型不确定性）: {np.mean(preds_std_avg):.4f} °C")
    print(f"集成预测间的标准差（反映模型差异）: {np.mean(preds_inter_std):.4f} °C")
