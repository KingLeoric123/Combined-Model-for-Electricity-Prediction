import pandas as pd
import numpy as np

def add_salt_pepper_noise(df, salt_prob=0.000001, pepper_prob=0.000001, exclude_cols=['date']):
    """
    向 DataFrame 的数值列添加椒盐噪声。
    
    参数:
        df: pandas DataFrame
        salt_prob: 每个元素被替换为最大值的概率（盐噪声）
        pepper_prob: 每个元素被替换为最小值的概率（椒噪声）
        exclude_cols: 不添加噪声的列名列表
    
    返回:
        添加噪声后的 DataFrame
    """
    df_noisy = df.copy()
    
    # 获取需要处理的数值列（排除指定列）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_process = [col for col in numeric_cols if col not in exclude_cols]
    
    for col in cols_to_process:
        series = df_noisy[col].copy()
        col_min = series.min()
        col_max = series.max()
        
        # 生成随机掩码
        rand_vals = np.random.random(size=len(series))
        
        # 盐噪声（替换为最大值）
        salt_mask = rand_vals < salt_prob
        # 椒噪声（替换为最小值）
        pepper_mask = (rand_vals >= salt_prob) & (rand_vals < salt_prob + pepper_prob)
        
        series[salt_mask] = col_max
        series[pepper_mask] = col_min
        
        df_noisy[col] = series
    
    return df_noisy


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 请修改为你的实际输入文件路径
    input_csv = r"C:\Users\Terry\Desktop\ABC\立项\2026-交通+教学\2026-统计建模大赛\数据集\原始数据集\ETTh2.csv"

    output_csv = r"C:\Users\Terry\Desktop\ABC\立项\2026-交通+教学\2026-统计建模大赛\数据集\用于测试模型稳健性的数据集\椒盐噪声\0.01noise_h2.csv"
    
    # 椒盐噪声比例（例如总噪声比例 = 5%，盐和椒各占一半）
    total_noise_ratio = 0.01   # 0.0005 的数据点被噪声污染
    salt_prob = total_noise_ratio / 2
    pepper_prob = total_noise_ratio / 2
    
    # 读取数据
    try:
        df_original = pd.read_csv(input_csv)
        print(f"成功读取 {input_csv}，形状: {df_original.shape}")
        print("列名:", df_original.columns.tolist())
    except FileNotFoundError:
        print(f"文件 {input_csv} 不存在，请检查路径。")
        exit(1)
    
    # 添加椒盐噪声
    df_noisy = add_salt_pepper_noise(df_original, salt_prob, pepper_prob, exclude_cols=['date'])
    
    # 保存结果
    df_noisy.to_csv(output_csv, index=False)
    print(f"噪声数据已保存至 {output_csv}")
    
    # 可选：输出前几行查看效果
    print("\n原始数据前3行:")
    print(df_original.head(3))
    print("\n加噪声后数据前3行:")
    print(df_noisy.head(3))
