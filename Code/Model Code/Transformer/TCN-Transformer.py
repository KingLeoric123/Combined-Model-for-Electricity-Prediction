read_file = r''
results = r''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# -------------------- 1. 数据准备（含滞后特征） --------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        """
        data: numpy array, shape (total_samples, n_features + 1)
              最后一列是目标值 (OT)，其余列是特征（包含原始特征和OT滞后特征）
        seq_len: 输入序列长度
        pred_len: 预测序列长度
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = data[:, :-1]  # 所有特征
        self.targets = data[:, -1:]   # 原始OT（未滞后）
        self.samples = []
        for i in range(len(self.features) - seq_len - pred_len + 1):
            x = self.features[i:i+seq_len]                      # (seq_len, n_features)
            y = self.targets[i+seq_len:i+seq_len+pred_len].flatten()  # (pred_len,)
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def load_and_prepare_data(csv_path, seq_len=96, pred_len=24, lag_len=96):
    """
    读取CSV，构造OT的滞后特征，划分训练/验证/测试集，并标准化。
    lag_len: 使用过去多少个OT值作为滞后特征（建议等于seq_len或更小）
    """
    df = pd.read_csv(csv_path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.drop('date', axis=1, inplace=True)

    # 原始特征列（不包含OT）
    raw_feature_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    target_col = 'OT'

    # 创建滞后特征：过去 lag_len 个时刻的 OT 值
    for lag in range(1, lag_len + 1):
        df[f'OT_lag{lag}'] = df[target_col].shift(lag)

    # 删除包含NaN的行（前lag_len行）
    df = df.dropna().reset_index(drop=True)

    # 特征列 = 原始特征 + 滞后特征
    feature_cols = raw_feature_cols + [f'OT_lag{lag}' for lag in range(1, lag_len+1)]
    data = df[feature_cols + [target_col]].values.astype(np.float32)

    total_len = len(data)
    train_end = int(total_len * 0.8)
    val_end = int(total_len * 0.9)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)

    train_dataset = TimeSeriesDataset(train_scaled, seq_len, pred_len)
    val_dataset = TimeSeriesDataset(val_scaled, seq_len, pred_len)
    test_dataset = TimeSeriesDataset(test_scaled, seq_len, pred_len)

    return train_dataset, val_dataset, test_dataset, scaler, len(feature_cols)


# -------------------- 2. TCN 模块 --------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp = Chomp1d(padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.relu(self.net(x) + self.residual(x))

class TCN(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        in_ch = input_dim
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.output_dim = num_channels[-1]

    def forward(self, x):
        # x: (B, L, D) -> 需要转换为 (B, D, L) 因为 Conv1d 期望 (batch, channels, length)
        x = x.permute(0, 2, 1)   # (B, D, L)
        out = self.net(x)        # (B, out_channels, L)
        return out.permute(0, 2, 1)  # 还原为 (B, L, out_channels)


# -------------------- 3. TimesNet 模块 --------------------
class TimesBlock(nn.Module):
    def __init__(self, seq_len, d_model, kernel_size=3, periods=[24, 12, 8]):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.kernel_size = kernel_size
        # 只保留能整除 seq_len 的周期
        self.periods = [p for p in periods if 1 < p < seq_len and seq_len % p == 0]
        if not self.periods:
            raise ValueError(f"None of the periods {periods} can divide seq_len {seq_len}")

        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(d_model),
            nn.ReLU()
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        out_list = []
        for period in self.periods:
            # 折叠为 2D: (B, L, D) -> (B, period, L//period, D) -> (B, D, period, L//period)
            x_2d = x.reshape(B, period, L//period, D).permute(0, 3, 1, 2)
            x_2d_conv = self.conv(x_2d)                     # (B, D, period, L//period)
            x_2d_pool = self.adaptive_pool(x_2d_conv)       # (B, D, period, 1)
            x_1d = x_2d_pool.squeeze(-1).permute(0, 2, 1)   # (B, period, D)
            # 线性插值回原始长度 L
            x_1d_interp = F.interpolate(x_1d.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
            out_list.append(x_1d_interp)
        # 多个周期的输出相加
        out = torch.stack(out_list, dim=0).sum(dim=0) if out_list else x
        return out

class TimesNet(nn.Module):
    def __init__(self, seq_len, d_model, num_blocks=2, kernel_size=3, periods=[24, 12, 8]):
        super().__init__()
        self.blocks = nn.ModuleList([
            TimesBlock(seq_len, d_model, kernel_size, periods) for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


# -------------------- 4. 组合模型 TCN + TimesNet --------------------
class TCNTimesNet(nn.Module):
    def __init__(self, input_dim, tcn_channels, seq_len, kernel_size=3, tcn_dropout=0.2,
                 timesnet_blocks=2, timesnet_periods=[24, 12, 8], timesnet_kernel=3,
                 pred_len=24):
        super().__init__()
        self.tcn = TCN(input_dim, tcn_channels, kernel_size, tcn_dropout)
        tcn_out_dim = tcn_channels[-1]
        self.timesnet = TimesNet(seq_len, tcn_out_dim, timesnet_blocks, timesnet_kernel, timesnet_periods)
        self.predictor = nn.Linear(tcn_out_dim, pred_len)
        self.pred_len = pred_len

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        tcn_out = self.tcn(x)               # (batch, seq_len, tcn_dim)
        times_out = self.timesnet(tcn_out)  # (batch, seq_len, tcn_dim)
        last_feat = times_out[:, -1, :]     # (batch, tcn_dim)
        pred = self.predictor(last_feat)    # (batch, pred_len)
        return pred.unsqueeze(-1)           # (batch, pred_len, 1)


# -------------------- 5. 训练与评估函数 --------------------
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)                     # (batch, pred_len, 1)
        loss = criterion(pred.squeeze(-1), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred.squeeze(-1), y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(dataloader.dataset)

def inverse_transform_predictions(pred_np, scaler, target_index=-1):
    """
    将标准化后的预测值（只有OT）反标准化为原始尺度。
    假设 scaler 是对完整 data（特征+目标）拟合的，目标列位于最后一列。
    """
    n_features = scaler.mean_.shape[0]
    dummy = np.zeros((pred_np.size, n_features))
    dummy[:, target_index] = pred_np.flatten()
    inv_dummy = scaler.inverse_transform(dummy)
    return inv_dummy[:, target_index].reshape(pred_np.shape)


# -------------------- 6. 主程序 --------------------
if __name__ == "__main__":
    csv_path = read_file
    seq_len = 96
    pred_len = 24
    lag_len = 96            # 滞后特征长度（建议与 seq_len 相同）
    batch_size = 64
    epochs = 50
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据（自动构造滞后特征）
    train_dataset, val_dataset, test_dataset, scaler, input_dim = load_and_prepare_data(
        csv_path, seq_len, pred_len, lag_len
    )
    print(f"特征维度（含滞后OT）: {input_dim}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型参数
    tcn_channels = [32, 32, 64]
    model = TCNTimesNet(
        input_dim=input_dim,
        tcn_channels=tcn_channels,
        seq_len=seq_len,
        kernel_size=3,
        tcn_dropout=0.2,
        timesnet_blocks=2,
        timesnet_periods=[24, 12, 8],    # 确保能被 seq_len=96 整除
        timesnet_kernel=3,
        pred_len=pred_len
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model_tcn_timesnet.pth")

    # 测试
    model.load_state_dict(torch.load("best_model_tcn_timesnet.pth"))
    test_loss = eval_epoch(model, test_loader, criterion, device)
    print(f"Test Loss (MSE): {test_loss:.6f}")

    # 保存预测示例
    model.eval()
    with torch.no_grad():
        x_sample, y_sample = next(iter(test_loader))
        x_sample = x_sample.to(device)
        pred_sample = model(x_sample).cpu().numpy()          # (batch, pred_len, 1)
        y_sample = y_sample.numpy()
        pred_original = inverse_transform_predictions(pred_sample.squeeze(-1), scaler)
        true_original = inverse_transform_predictions(y_sample, scaler)

        results_df = pd.DataFrame({
            'true_value': true_original.flatten(),
            'predicted_value': pred_original.flatten(),
            'residual': (true_original - pred_original).flatten()
        })
        results_df.to_csv(results, index=False)
        print("预测结果已保存为 TCN_TimesNet_predictions.csv")
