import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def detect_outliers_by_svd(df, numeric_cols, k=3, threshold_method='iqr', percentile=95):
    # 提取数值矩阵
    X = df[numeric_cols].values.astype(float)#提取要分析的数据列表
    
    # 标准化（中心化 + 缩放）—— SVD 对尺度敏感
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # SVD
    U, s, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    
    # 低秩近似：仅保留前 k 个奇异值
    s_k = np.zeros_like(s)
    s_k[:k] = s[:k]
    S_k = np.diag(s_k)
    X_reconstructed = U @ S_k @ Vt
    
    # 计算每行的重构误差（均方根误差 RMSE）
    errors = np.sqrt(np.mean((X_scaled - X_reconstructed) ** 2, axis=1))
    
    # 根据阈值方法标记异常
    if threshold_method == 'iqr':
        Q1 = np.percentile(errors, 25)
        Q3 = np.percentile(errors, 75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        outlier_mask = errors > upper_bound
    elif threshold_method == 'percentile':
        threshold = np.percentile(errors, percentile)
        outlier_mask = errors > threshold
    else:
        raise ValueError("threshold_method 必须是 'iqr' 或 'percentile'")
    
    return outlier_mask, errors

if __name__ == "__main__":
    # 配置路径
    input_csv = r"C:\Users\Terry\Desktop\ABC\立项\2026-交通+教学\2026-统计建模大赛\数据集\用于测试模型稳健性的数据集\椒盐噪声\0.005noise_h2.csv"   # 已经添加椒盐噪声的文件
    output_csv = r"C:\Users\Terry\Desktop\ABC\立项\2026-交通+教学\2026-统计建模大赛\数据集\用于测试模型稳健性的数据集\去除了椒盐噪声的数据集\cnoise_0.005h2.csv" # 删除异常值后的文件
    
    # 数值列名
    numeric_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    date_col = 'date'
    
    # 读取数据
    df = pd.read_csv(input_csv)
    print(f"原始数据形状: {df.shape}")
    
    # 确保 date 列存在且为 object/datetime，不做数值处理
    # 检查是否存在缺失值
    if df[numeric_cols].isnull().any().any():
        print("警告：数据中存在缺失值，请先处理缺失值（例如删除或插补）")
        #删除含有缺失值的行
        df = df.dropna(subset=numeric_cols)
        print(f"删除缺失值后形状: {df.shape}")
    
    # 检测异常行
    outlier_mask, errors = detect_outliers_by_svd(
        df, numeric_cols, 
        k=3,
        threshold_method='iqr',  # 或 'percentile'
        percentile=95
    )
    
    n_outliers = outlier_mask.sum()
    print(f"检测到异常行数: {n_outliers} ({n_outliers/len(df)*100:.2f}%)")
    
    # 删除异常行
    df_cleaned = df[~outlier_mask].copy()
    print(f"清洗后数据形状: {df_cleaned.shape}")
    
    # 可选：保存清洗后的数据
    df_cleaned.to_csv(output_csv, index=False)
    print(f"已保存至 {output_csv}")
    
    # 可选：显示误差统计
    print(f"\n重构误差统计: min={errors.min():.4f}, max={errors.max():.4f}, mean={errors.mean():.4f}")
