
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_FOLDER = PROJECT_ROOT / "reports"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_FOLDER / "svm_confusion_matrix.png"

# نتایج واقعی ارزیابی مدل روی مجموعه آزمون
cm = np.array([
    [166, 34],
    [15, 585]
])

labels = ["Human", "AI-generated"]

fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(cm, cmap="Blues")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("SVM Confusion Matrix - Test Dataset")

for i in range(2):
    for j in range(2):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color=color,
            fontsize=16,
            fontweight="bold"
        )

fig.colorbar(image, ax=ax, label="Number of samples")

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
plt.close(fig)

print("Confusion matrix chart saved successfully!")
print(f"File: {OUTPUT_FILE}")
