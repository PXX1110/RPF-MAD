import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Data
data_with_lr = [
    (0.1, 0.05, 96, 15), (0.1, 0.01, 94, 15), (0.1, 0.005, 94, 15), (0.1, 0.0025, 94, 14), (0.1, 0.001, 96, 15),
    (0.01, 0.05, 82, 26), (0.01, 0.01, 82, 28), (0.01, 0.005, 83, 28), (0.01, 0.0025, 83, 27), (0.01, 0.001, 82, 28),
    (0.005, 0.05, 83, 26), (0.005, 0.01, 79, 29), (0.005, 0.005, 78, 30), (0.005, 0.0025, 79, 29), (0.005, 0.001, 79, 29),
    (0.0025, 0.05, 80, 26), (0.0025, 0.01, 77, 31), (0.0025, 0.005, 76, 31), (0.0025, 0.0025, 76, 29), (0.0025, 0.001, 74, 28),
    (0.001, 0.05, 76, 30), (0.001, 0.01, 75, 32), (0.001, 0.005, 74, 31), (0.001, 0.0025, 73, 31), (0.001, 0.001, 73, 31)
]

# Extract data
accuracy = np.array([item[2] for item in data_with_lr])
error = np.array([item[3] for item in data_with_lr])
lr_e = [item[0] for item in data_with_lr]
lr_f = [item[1] for item in data_with_lr]

# Compute Pareto Front
def pareto_frontier(Xs, Ys, maxX=True, maxY=False):
    sorted_list = sorted([[Xs[i], Ys[i], lr_e[i], lr_f[i]] for i in range(len(Xs))], reverse=maxX)
    pareto_front = [sorted_list[0]]
    for pair in sorted_list[1:]:
        if maxY:
            if pair[1] < pareto_front[-1][1]:
                pareto_front.append(pair)
        else:
            if pair[1] > pareto_front[-1][1]:
                pareto_front.append(pair)
    return np.array(pareto_front)

pareto_points = pareto_frontier(accuracy, error, maxX=True, maxY=False)

# Extract Pareto front values
pareto_x, pareto_y, pareto_lr_e, pareto_lr_f = pareto_points[:, 0], pareto_points[:, 1], pareto_points[:, 2], pareto_points[:, 3]
# 颜色渐变（用于 Pareto Front）
colors = np.linspace(0.2, 1, len(pareto_x))  # 颜色渐变参数
# Create figure
plt.figure(figsize=(6, 4))

# 画出所有数据点（非 Pareto 点）
plt.scatter(accuracy, error, label="All Points", color="lightgray", alpha=1, s=40)

# 画出 Pareto Front（红色渐变）
plt.scatter(pareto_x, pareto_y, c=colors, cmap="Reds", s=100, label="Pareto Points", edgecolors="black", linewidth=1.2)

# 画出 Pareto Front 线条（使用渐变色）
for i in range(len(pareto_x) - 1):
    plt.plot([pareto_x[i], pareto_x[i+1]], [pareto_y[i], pareto_y[i+1]], color="darkred", linewidth=2.5, alpha=1)

# **优化文本标注**（数字居中）
for i, (x, y) in enumerate(zip(pareto_x, pareto_y)):
    plt.annotate(str(i + 1), (x, y),
                 textcoords="offset points", xytext=(0, 0), ha='center', va='center',
                 fontsize=10, fontweight="bold", color="yellow",
                 bbox=dict(boxstyle="circle,pad=0.3", fc="darkred", ec="black", lw=1.2))

plt.xlabel('Accuracy', fontsize=12)
plt.ylabel('Error Rate', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
# 设置图例
# plt.legend(frameon=False, loc='upper right', fontsize=10)
plt.legend(framealpha=1)

# **优化布局**
# plt.subplots_adjust(left=0.15, right=0.95, top=0.90, bottom=0.15)

# # 保存为 SVG（适合网页、论文）
# plt.savefig('pareto_front_optimized.svg', format='svg', bbox_inches='tight')

# # 保存为 PDF（适合论文、打印）
# plt.savefig('pareto_front_optimized.pdf', format='pdf', bbox_inches='tight')

# 保存为 EPS（适合 LaTeX）
plt.savefig('pareto_front_optimize.eps', format='eps', bbox_inches='tight')

# 显示
plt.show()
