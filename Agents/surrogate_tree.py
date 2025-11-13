# agents/surrogate_tree.py
from sklearn.tree import DecisionTreeClassifier, export_text
import numpy as np
import joblib
import os

# Load state-action data (allow_pickle=True for object arrays)
states = np.load("logs/states.npy", allow_pickle=True)
actions = np.load("logs/actions.npy")

# Pad states to same length
max_len = max(len(s) for s in states)
X = np.array([list(s) + [0]*(max_len - len(s)) for s in states])
y = actions

# Fit decision tree surrogate
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X, y)

# Optional: visualize textual rules
tree_rules = export_text(tree, feature_names=[f"P{i+1}" for i in range(X.shape[1])])
print("Surrogate Decision Tree Rules:\n", tree_rules)

# Save tree for API usage
os.makedirs("models", exist_ok=True)
joblib.dump(tree, "models/surrogate_tree.pkl")
print("Surrogate tree saved to models/surrogate_tree.pkl")
