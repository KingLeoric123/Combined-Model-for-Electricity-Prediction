read_file = r"C:\Users\Terry\Desktop\ABC\立项\2026-交通+教学\2026-统计建模大赛\数据集\原始数据集\ETTm1.csv"
results_df_path = None

import shap
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
# ====================== 新增：导入 SHAP 库 ======================
import shap

# ... 前面的导入和类定义保持不变 ...

# ====================== 新增：SHAP 分析函数 ======================
def shap_analysis(model, background_data, test_data, feature_names, seq_len, device='cuda'):
    """
    使用 SamplingExplainer 计算 SHAP 值，并汇总特征重要性。
    
    参数：
        model: 训练好的 Transformer 模型
        background_data: numpy 数组，形状 (n_background, seq_len, n_features)
        test_data: numpy 数组，形状 (n_test, seq_len, n_features)
        feature_names: 特征名称列表
        seq_len: 序列长度
        device: 计算设备
    返回：
        shap_values: SHAP 值数组，形状 (n_test, seq_len * n_features)
        feature_importance: 按特征汇总的重要性字典
    """
    # 将模型切换到 CPU 并设置为评估模式（SHAP 在 CPU 上更稳定）
    model = model.to('cuda')
    model.eval()
    
    # 定义预测函数，输入为二维数组 (样本数, seq_len * n_features)
    def predict_func(x):
        # 重塑为 (batch, seq_len, n_features)
        n_samples = x.shape[0]
        x_reshaped = x.reshape(n_samples, seq_len, -1)
        x_tensor = torch.tensor(x_reshaped, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred = model(x_tensor).cpu().numpy()
        return pred.flatten()
    
    # 准备背景数据和测试数据（展平为二维）
    n_features = background_data.shape[-1]
    background_flat = background_data.reshape(background_data.shape[0], -1)
    test_flat = test_data.reshape(test_data.shape[0], -1)
    
    print("正在创建 SHAP SamplingExplainer（这可能需要几分钟）...")
    explainer = shap.SamplingExplainer(predict_func, background_flat)
    
    print("正在计算 SHAP 值...")
    shap_values = explainer.shap_values(test_flat)
    
    # 汇总特征重要性（按原始特征求和，忽略时间步）
    # shap_values 形状: (n_test, seq_len * n_features)
    shap_reshaped = shap_values.reshape(-1, seq_len, n_features)
    
    # 按特征求和得到每个特征的总体重要性
    feature_importance = {}
    for i, name in enumerate(feature_names):
        # 对测试样本和时间步求和
        total_impact = np.sum(np.abs(shap_reshaped[:, :, i]))
        feature_importance[name] = total_impact
    
    # 归一化为百分比
    total = sum(feature_importance.values())
    for name in feature_importance:
        feature_importance[name] = feature_importance[name] / total * 100
    
    return shap_values, feature_importance


# ====================== 主程序修改部分 ======================
if __name__ == '__main__':

    all_preds = []
    all_trues = None
    best_model_state = None       # 保存最佳模型的 state_dict
    best_scaler_X = None
    best_scaler_y = None
    best_feature_cols = None
    best_seq_len = seq_len = 24

    
    # -------------------- 数据加载与预处理（与之前相同） --------------------
    df = pd.read_csv(read_file, parse_dates=['date'])
    base_features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    target_col = 'OT'
    feature_cols = base_features + [target_col]
    
    df = df.dropna(subset=feature_cols + [target_col])
    df = df.sort_values('date')
    
    train_ratio = 0.7
    val_ratio = 0.1
    n_total = len(df)
    train_end = int(train_ratio * n_total)
    val_end = train_end + int(val_ratio * n_total)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
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
    
    X_train, y_train = create_sequences(train_df_scaled, target_col, feature_cols, seq_len, pred_len=1)
    X_val, y_val = create_sequences(val_df_scaled, target_col, feature_cols, seq_len, pred_len=1)
    X_test, y_test = create_sequences(test_df_scaled, target_col, feature_cols, seq_len, pred_len=1)
    
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    input_dim = len(feature_cols)
    model = TransformerTemperaturePredictor(
        input_dim=input_dim, d_model=64, nhead=4,
        num_layers=3, dim_feedforward=128, dropout=0.1
    )
    
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, epochs=1, lr=1e-3,
        device=device, patience=10
    )
    
    # SHAP 特征重要性分析
    print("\n========== 开始 SHAP 特征重要性分析 ==========")
    
   
    model_shap = TransformerTemperaturePredictor(
        input_dim=len(feature_cols), d_model=64, nhead=4,
        num_layers=3, dim_feedforward=128, dropout=0.1
    )
    
    # sampling SHAP
    np.random.seed(42)
    background_indices = np.random.choice(len(X_train), size=min(100, len(X_train)), replace=False)
    background_data = X_train[background_indices]  # 形状 (n_bg, seq_len, n_features)
    
    # 准备测试数据：取测试集的前 100 个样本用于解释（可根据计算资源调整）
    n_test_shap = min(100, len(X_test))
    test_shap_data = X_test[:n_test_shap]
    
    # 调用 SHAP 分析函数
    shap_values, feature_importance = shap_analysis(
        model=model_shap,
        background_data=background_data,
        test_data=test_shap_data,
        feature_names=best_feature_cols,
        seq_len=seq_len,
        device=device
    )
    
    # 输出特征重要性排序
    print("\n特征重要性（基于 SHAP 绝对值总和，归一化百分比）：")
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for name, imp in sorted_importance:
        print(f"  {name}: {imp:.2f}%")
  
    # 保存特征重要性 CSV
    importance_df = pd.DataFrame(list(feature_importance.items()), columns=['Feature', 'Importance(%)'])
    importance_df = importance_df.sort_values('Importance(%)', ascending=False)
    importance_path = r"C:\Users\Terry\Desktop\ABC\立项\2026-交通+教学\2026-统计建模大赛\数据集\用于绘图的数据集\feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"特征重要性已保存至: {importance_path}")
    
    # 简单可视化（可选）
    plt.figure(figsize=(10, 5))
    plt.barh(importance_df['Feature'], importance_df['Importance(%)'])
    plt.xlabel('Importance (%)')
    plt.title('Feature Importance based on SHAP')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(r"C:\Users\Terry\Desktop\ABC\立项\2026-交通+教学\2026-统计建模大赛\示意图\shap_importance.png")
    plt.show()
