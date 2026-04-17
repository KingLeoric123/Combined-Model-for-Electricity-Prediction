read_file = r"C:\Users\Terry\Desktop\ABC\立项\2026-交通+教学\2026-统计建模大赛\数据集\原始数据集\ETTm1.csv"
results_df_path = None

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x[:, -1, :]
        out = self.output_proj(x)
        return out

# ====================== 3. 训练与评估函数 ======================
def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3, device='cuda', patience=50):
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

def evaluate_model(model, test_loader, device='cuda', scaler_y=None):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch).cpu().numpy()
            preds.append(pred)
            trues.append(y_batch.numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    if scaler_y is not None:
        trues = scaler_y.inverse_transform(trues.reshape(-1,1)).flatten()
        preds = scaler_y.inverse_transform(preds.reshape(-1,1)).flatten()
    
    rmse = np.sqrt(np.mean((preds - trues)**2))
    mae = np.mean(np.abs(preds - trues))
    ss_res = np.sum((trues - preds)**2)
    ss_tot = np.sum((trues - np.mean(trues))**2)
    r2 = 1-(ss_res/ss_tot) if ss_tot !=0 else 0.0
    
    print(f'Test RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')
    return trues, preds, rmse, mae, r2

#开启Monte-Carlo Dropout


# ====================== 4. 主程序（加载真实数据，运行三遍取平均） ======================
if __name__ == '__main__':
    n_runs = 3                     # 运行次数
    all_preds = []                 # 存储每次运行的预测值
    all_trues = None               # 真实值（每次相同）
    
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
        
        # -------------------- 测试评估 --------------------
        rmse = np.sqrt(np.mean((preds_mean - trues)**2))
        mae = np.mean(np.abs(preds_mean - trues))
        ss_res = np.sum((trues - preds_mean)**2)
        ss_tot = np.sum((trues - np.mean(trues))**2)
        r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0
        # 保存真实值（第一次运行后不再改变）
        if all_trues is None:
            all_trues = trues
        all_preds.append(preds)
        
        # 可选：预测下一时刻油温（使用最后一次模型）
        if run == n_runs - 1:
            last_seq = test_df[feature_cols].iloc[-seq_len:].values
            def predict_future(model, last_sequence, scaler_X, scaler_y, device='cuda'):
                model.eval()
                last_seq_scaled = scaler_X.transform(last_sequence)
                input_tensor = torch.tensor(last_seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_scaled = model(input_tensor).cpu().numpy()
                pred = scaler_y.inverse_transform(pred_scaled)
                return pred[0,0]
            next_temp = predict_future(model, last_seq, scaler_X, scaler_y, device)
            print(f"基于最后{seq_len}个历史数据，预测下一时刻油温: {next_temp:.2f} °C")
    
    # ========== 三遍运行结束，计算平均预测值及最终指标 ==========
    preds_avg = np.mean(all_preds, axis=0)
    
    # 使用平均预测值计算拟合优度
    rmse_avg = np.sqrt(np.mean((preds_avg - all_trues)**2))
    mae_avg = np.mean(np.abs(preds_avg - all_trues))
    ss_res = np.sum((all_trues - preds_avg)**2)
    ss_tot = np.sum((all_trues - np.mean(all_trues))**2)
    r2_avg = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0
    
    print("\n========== 最终集成结果（三次预测平均） ==========")
    print(f"RMSE (avg pred): {rmse_avg:.4f}")
    print(f"MAE  (avg pred): {mae_avg:.4f}")
    print(f"R2   (avg pred): {r2_avg:.4f}")
    
    # 在最终保存CSV时加入标准差列
    if run == n_runs - 1:   # 最后一次运行保存包含不确定性的结果
        results_df = pd.DataFrame({
            'true_value': all_trues,
            'predicted_mean': preds_avg,          # 三次运行均值的平均（如需）
            'predicted_std_last_run': preds_std    # 最后一次运行的预测标准差
        })
    else:
        results_df = pd.DataFrame({
            'true_value': all_trues,
            'predicted_value_avg': preds_avg
        })
