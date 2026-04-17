#Pyhon遵循左闭右开原则
#改进：数组填充
#注意 Warning：数据是经过微调的，某些数据中需要删除几列让x轴和y轴对齐。这些改动本身是不影响数据的真实性的
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams['font.family'] = 'serif' #设置字体
plt.rcParams['font.serif'] = ['Times New Roman'] #设置英文字体
plt.rcParams['font.size'] = 12 #设置字体大小
plt.rcParams['axes.linewidth'] = 1.0   # 坐标轴线宽
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

df = pd.read_csv(r"C:\Users\Terry\Desktop\ABC\立项\2026-交通+教学\2026-统计建模大赛\数据集\用于绘图的数据集\TCN-Transformer\m1.csv")
y2 = df['true_value']
y3 = df['predicted_value']
#y4 = df['residual']
#y5 = df['MUFL']
#y6 = df['LULL']
#y7 = df['LUFL']
#y1= df['OT']
x= range(len(y2))

#需要事先将y的值和x的值在同一维度上对齐，不要出现缺元素的情况

y2_padded = np.pad(y2, (len(x)-len(y2),0), constant_values = np.nan)
y3_padded = np.pad(y3, (len(x)-len(y2),0), constant_values = np.nan)
#y4_padded = np.pad(y4, (len(x)-len(y4),0), constant_values = np.nan)
#y5_padded = np.pad(y5, (len(x)-len(y5),0), constant_values = np.nan)
#y6_padded = np.pad(y6, ( len(x)-len(y6),0), constant_values = np.nan)
#y7_padded = np.pad(y7, (len(x)-len(y7),0), constant_values = np.nan)
# 创建图形
fig, ax = plt.subplots(figsize=(6, 3.6))

# 绘制曲线
ax.plot(x, y2, '#1E88E5', linewidth=1, label='True_Value')#深海蓝-真实值线条颜色
ax.plot(x, y2_padded, '#E67E22', alpha = 1, linewidth=0.5, label='TCN-Transformer_Predicted_Value')#橙红：LightGBM的预测值
ax.plot(x, y3, '#3ECC71', alpha = 0.5, linewidth=0.5, label='LSTM_Predicted_Value')#翠绿：LSTM的预测值
#ax.plot(x, y4_padded, '#9B59B6', alpha = 0.5, linewidth=0.5, label='XGBoost_Predicted_Value')#紫罗兰：XGBoost的预测值
#ax.plot(x, y5_padded, '#FF00FF', linewidth = 0.5,alpha=0.5, label = 'Transformer_Predicted_Value')#品红色:Transformer的预测值
#ax.plot(x, y6_padded, '#4DBBD5', linewidth = 0.5,alpha=0.5, label = 'ARIMA_Pridicted_Value')#青绿：ARIMA的预测值
#ax.plot(x, y7_padded, '#CCBB44', linewidth = 0.5,alpha=0.5, label = 'SVD-Transformer_Predicted_Value')#芥末黄：SVD-Transformer的预测值
ax.legend(loc = 'upper left')
# 坐标轴标签
ax.set_xlabel('TimeStep', fontsize=12)
ax.set_ylabel('Oil Tenperature(OT)', fontsize=12)

# 隐藏上边和右边边框（经典学术干净风格）
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 图例（干净位置）
ax.legend(frameon=False, loc='upper left', fontsize = 7)

# 添加网格（可选，浅灰色增加可读性）
ax.grid(True, linestyle='--', alpha=0.4)

# 自动调整布局
plt.tight_layout()

# 保存为矢量 PDF（无限放大不失真）
#plt.savefig('academic_plot.pdf', format='pdf', dpi=300)
# 也可以保存为 SVG
plt.savefig(r'C:\Users\Terry\Desktop\ABC\立项\2026-交通+教学\2026-统计建模大赛\示意图\Cnoise.svg', format='svg')

#显示
plt.show()
