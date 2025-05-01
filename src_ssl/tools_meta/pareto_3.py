import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Ellipse
# 修改样式配置部分为：
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Arial'],
    'text.usetex': False,
    'savefig.dpi': 300,
    'figure.figsize': (8, 6),
    'axes.edgecolor': '0.3'
})

# 原始数据矩阵
raw_data = [
    (0.1, 0.05, 96, 15), (0.1, 0.01, 94, 15), (0.1, 0.005, 94, 15),
    (0.1, 0.0025, 94, 14), (0.1, 0.001, 96, 15),(0.01, 0.05, 82, 26),
    (0.01, 0.01, 82, 28), (0.01, 0.005, 83, 28), (0.01, 0.0025, 83, 27),
    (0.01, 0.001, 82, 28),(0.005, 0.05, 83, 26), (0.005, 0.01, 79, 29),
    (0.005, 0.005, 78, 30), (0.005, 0.0025, 79, 29),(0.005, 0.001, 79, 29),
    (0.0025, 0.05, 80, 26), (0.0025, 0.01, 77, 31), (0.0025, 0.005, 76, 31),
    (0.0025, 0.0025, 76, 29), (0.0025, 0.001, 74, 28),(0.001, 0.05, 76, 30),
    (0.001, 0.01, 75, 32), (0.001, 0.005, 74, 31), (0.001, 0.0025, 73, 31),
    (0.001, 0.001, 73, 31)
]

# 数据预处理
data_matrix = np.array(raw_data)
lr_encoder = data_matrix[:, 0]
lr_decoder = data_matrix[:, 1]
semantic_acc = data_matrix[:, 2]
action_acc = data_matrix[:, 3]

# 目标配置点
highlight_index = np.where((semantic_acc == 83) & (action_acc == 28))[0][0]

# 帕累托前沿计算函数
def find_pareto_front(scores):
    pareto_mask = np.ones(scores.shape[0], dtype=bool)
    for i in range(len(scores)):
        if pareto_mask[i]:
            pareto_mask[pareto_mask] = np.any(scores[pareto_mask] > scores[i], axis=1)
            pareto_mask[i] = True
    return np.where(pareto_mask)[0]

pareto_points = find_pareto_front(np.column_stack((semantic_acc, action_acc)))

# 可视化引擎初始化
fig, ax = plt.subplots()

# 颜色映射设置
color_norm = mcolors.LogNorm(vmin=lr_encoder.min(), vmax=lr_encoder.max())
scatter_plot = ax.scatter(
    semantic_acc, 
    action_acc,
    c=lr_encoder,
    s=lr_decoder*1200,
    cmap='coolwarm',
    norm=color_norm,
    edgecolor='white',
    linewidth=1.2,
    alpha=0.85
)

# 绘制帕累托前沿
sorted_indices = np.argsort(semantic_acc[pareto_points])
ax.plot(
    semantic_acc[pareto_points][sorted_indices], 
    action_acc[pareto_points][sorted_indices],
    color='#2C3E50',
    linestyle='-.',
    linewidth=2.5,
    marker='',
    alpha=0.7,
    label='Pareto Frontier'
)

# 高亮最优配置
ax.scatter(
    semantic_acc[highlight_index],
    action_acc[highlight_index],
    s=350,
    facecolors='none', 
    edgecolors='#E74C3C',
    linewidth=2.5,
    zorder=10,
    label='Optimal Setting'
)

# 添加特殊标注
annotation_box = Ellipse(
    (82, 27.3), 
    width=8, 
    height=3.5, 
    angle=15,
    edgecolor='#3498DB',
    linestyle='--',
    linewidth=1.5,
    fill=False
)
ax.add_patch(annotation_box)

# 颜色图例配置
cbar = plt.colorbar(scatter_plot, pad=0.02)
cbar.set_label('Encoder Learning Rate', rotation=270, labelpad=18)

# 尺寸图例配置
size_legend = [
    (0.05, 'LR=0.05'),
    (0.01, 'LR=0.01'), 
    (0.005, 'LR=0.005')
]
for sz, lb in size_legend:
    ax.scatter([], [], s=sz*1200, 
               edgecolor='gray', 
               facecolor='none',
               linewidth=1.2,
               label=lb)
lgnd = ax.legend(
    title='Decoder LR', 
    loc='upper left',
    bbox_to_anchor=(1.02, 1),
    frameon=True,
    borderpad=1.5
)

# 坐标系参数调整
ax.set(
    xlim=(72, 97),
    ylim=(13, 33),
    xlabel='Semantic Accuracy (%)',
    ylabel='Action Accuracy (%)'
)
ax.set_title('Multi-Objective Optimization Landscape', pad=18)

# 移除网格线并优化渲染
ax.grid(False)
plt.setp(ax.get_xticklines(), visible=True)
plt.setp(ax.get_yticklines(), visible=True)

# 保存和分析
plt.tight_layout()
plt.savefig('optimization_analysis.eps', format='eps', bbox_inches='tight')
plt.close()
