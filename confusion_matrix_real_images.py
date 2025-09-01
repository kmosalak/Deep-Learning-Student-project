import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Class labels
labels = [
    'veritasium', 'crashcourse', 'MKBHD', 'Vox', 'GrahamStephan',
    'jacksepticeye', 'chloeting', 'BingingWithBabish', 'CollegeHumor', '5minutecrafts'
]

# Support values from Epoch 10
support = {
    'veritasium': 88,
    'crashcourse': 81,
    'MKBHD': 77,
    'Vox': 86,
    'GrahamStephan': 89,
    'jacksepticeye': 90,
    'chloeting': 80,
    'BingingWithBabish': 67,
    'CollegeHumor': 70,
    '5minutecrafts': 72
}

# Recall values from Epoch 10
recall = {
    'veritasium': 0.36,
    'crashcourse': 0.90,
    'MKBHD': 0.45,
    'Vox': 0.65,
    'GrahamStephan': 0.65,
    'jacksepticeye': 0.50,
    'chloeting': 0.75,
    'BingingWithBabish': 0.67,
    'CollegeHumor': 0.74,
    '5minutecrafts': 0.76
}

# Initialize confusion matrix
conf_matrix = np.zeros((10, 10), dtype=int)

# Populate diagonal with TP (recall * support), remaining distributed uniformly across other classes
for i, label in enumerate(labels):
    tp = int(round(recall[label] * support[label]))
    fn = support[label] - tp
    conf_matrix[i, i] = tp
    
    # Distribute FN across other classes
    distribute_fn = fn // 9
    for j in range(10):
        if i != j:
            conf_matrix[i, j] = distribute_fn
    leftover = fn - distribute_fn * 9
    for j in range(leftover):
        idx = (i + j + 1) % 10
        if idx != i:
            conf_matrix[i, idx] += 1

# Create a heatmap
plt.figure(figsize=(10, 8))
df_cm = pd.DataFrame(conf_matrix, index=labels, columns=labels)
sns.heatmap(df_cm, annot=True, fmt="d", cmap="YlGnBu", cbar=True)
plt.title("Estimated Confusion Matrix Heatmap (Epoch 10)")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.tight_layout()
plt.show()

# Save as PNG
plt.savefig("estimated_confusion_matrix_epoch10.png", dpi=300)
plt.close()
