read_file = r"C:\Users\Terry\Desktop\ABC\立项\2026-交通+教学\2026-统计建模大赛\数据集\原始数据集\ETTm1.csv"
results_df_path = r"C:\Users\Terry\Desktop\ABC\立项\2026-交通+教学\2026-统计建模大赛\数据集\用于绘图的数据集\LSTM\add.csv"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

# 尝试导入 auto_arima，如果没有则使用固定阶数
try:
    from pmdarima import auto_arima
    AUTO_ARIMA_AVAILABLE = True
except ImportError:
    AUTO_ARIMA_AVAILABLE = False
    print("pmdarima 未安装，将使用固定阶数 ARIMA(5,1,0)")

# ====================== 1. 数据加载与预处理 ======================
data_path = r"C:\Users\Terry\Desktop\ABC\立项\2026-交通+教学\2026-统计建模大赛\数据集\原始数据集\ETTm1.csv"
data_path = read_file
df = pd.read_csv(data_path, parse_dates=['date'])
target_col = 'OT'

# 按时间排序
df = df.sort_values('date')
# 去除缺失值
df = df.dropna(subset=[target_col])

# 划分训练/验证/测试集（与 Transformer 相同比例）
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

# ARIMA 只需要单变量序列，合并训练+验证作为训练数据（让模型看到更多模式）
train_series = pd.concat([train_df[target_col], val_df[target_col]], ignore_index=True)
test_series = test_df[target_col].reset_index(drop=True)

# ====================== 2. 自动或手动确定 ARIMA 阶数 ======================
if AUTO_ARIMA_AVAILABLE:
    print("正在使用 auto_arima 搜索最佳参数...")
    # 搜索空间：p=0~7, d=0~2, q=0~7，季节性不考虑（数据无明显周期）
    auto_model = auto_arima(
        train_series,
        start_p=0, max_p=7,
        start_d=0, max_d=2,
        start_q=0, max_q=7,
        seasonal=False,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        n_fits=50
    )
    order = auto_model.order
    print(f"自动选择的最佳阶数: ARIMA{order}")
else:
    order = (5, 1, 0)   # 默认阶数
    print(f"使用固定阶数: ARIMA{order}")

# ====================== 3. 滚动预测函数 ======================
def rolling_arima_forecast(series, order, window_size=24):
    """
    使用固定窗口滚动预测：每步用最近 window_size 个真实值拟合 ARIMA，预测下一个点
    series : 完整的测试序列（真实值）
    order  : (p,d,q)
    window_size : 每次拟合使用的历史观测数
    返回预测值数组（长度 = len(series) - window_size）
    """
    n = len(series)
    predictions = []
    for t in range(window_size, n):
        # 取历史窗口的真实值
        hist = series.iloc[t-window_size:t] if isinstance(series, pd.Series) else series[t-window_size:t]
        model = ARIMA(hist, order=order)
        fitted = model.fit()
        pred = fitted.forecast(steps=1).iloc[0]
        predictions.append(pred)
    return np.array(predictions)

# 窗口长度与 Transformer 的 seq_len 一致
seq_len = 24
print(f"\n使用滚动窗口大小: {seq_len}")

# 在测试集上滚动预测
test_series = test_series.reset_index(drop=True)  # 确保索引从0开始
preds = rolling_arima_forecast(test_series, order, window_size=seq_len)

# 真实值对应的是窗口之后的点（从索引 seq_len 开始）
trues = test_series.iloc[seq_len:].values

# ====================== 4. 评估指标 ======================
rmse = np.sqrt(mean_squared_error(trues, preds))
mae = mean_absolute_error(trues, preds)
r2 = r2_score(trues, preds)

print("\n========== ARIMA 预测结果 ==========")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R2  : {r2:.4f}")

# ====================== 5. 保存预测结果 ======================
results_df = pd.DataFrame({
    'true_value': trues,
    'predicted_value_arima': preds,
    'residual': trues - preds
})
output_path = results_df_path
results_df.to_csv(output_path, index=False)
print(f"\n预测结果已保存至：{output_path}")
