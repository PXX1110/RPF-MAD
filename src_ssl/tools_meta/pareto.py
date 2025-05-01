import numpy as np
import matplotlib.pyplot as plt

# Data from the table with corresponding LR_e and LR_f (accuracy/error)
lr_e_values = [0.1, 0.01, 0.005, 0.0025, 0.001]
lr_f_values = [0.05, 0.01, 0.005, 0.0025, 0.001, 0.0001]

# Accuracy/error data with corresponding LR_e and LR_f indices
data_with_lr = [
    (0.1, 0.05, 96, 15), (0.1, 0.01, 94, 15), (0.1, 0.005, 94, 15), (0.1, 0.0025, 94, 14), (0.1, 0.001, 96, 15),
    (0.01, 0.05, 82, 26), (0.01, 0.01, 82, 28), (0.01, 0.005, 83, 28), (0.01, 0.0025, 83, 27), (0.01, 0.001, 82, 28),
    (0.005, 0.05, 83, 26), (0.005, 0.01, 79, 29), (0.005, 0.005, 78, 30), (0.005, 0.0025, 79, 29), (0.005, 0.001, 79, 29),
    (0.0025, 0.05, 80, 26), (0.0025, 0.01, 77, 31), (0.0025, 0.005, 76, 31), (0.0025, 0.0025, 76, 29), (0.0025, 0.001, 74, 28),
    (0.001, 0.05, 76, 30), (0.001, 0.01, 75, 32), (0.001, 0.005, 74, 31), (0.001, 0.0025, 73, 31), (0.001, 0.001, 73, 31)
]

# Extract accuracy, error, LR_e, and LR_f
accuracy = np.array([item[2] for item in data_with_lr])
error = np.array([item[3] for item in data_with_lr])
lr_e = [item[0] for item in data_with_lr]
lr_f = [item[1] for item in data_with_lr]

# Function to find Pareto front
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

# Calculate Pareto front
pareto_points = pareto_frontier(accuracy, error, maxX=True, maxY=False)

# Plot all points
plt.scatter(accuracy, error, label="All Points", color="blue")

# Highlight Pareto front points
plt.plot(pareto_points[:, 0], pareto_points[:, 1], label="Pareto Front", color="red", linewidth=2, marker="o")

# Annotate points with LR_e and LR_f combinations
for i in range(len(pareto_points)):
    plt.annotate(f"(e={pareto_points[i, 2]}, f={pareto_points[i, 3]})",
                 (pareto_points[i, 0], pareto_points[i, 1]),
                 textcoords="offset points", xytext=(0, 5), ha='center')

plt.xlabel('Accuracy')
plt.ylabel('Error Rate')
plt.title('Pareto Front for Accuracy vs Error Rate with LR_e and LR_f')
plt.legend()
plt.grid(True)

# Save the figure as an image
plt.savefig('pareto_front_with_lr.png')

# Show the plot
plt.show()
