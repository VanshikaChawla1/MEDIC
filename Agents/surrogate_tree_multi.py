import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
import joblib

# Load logs
states = np.load("logs/multi_states.npy", allow_pickle=True)
actions = np.load("logs/multi_actions.npy", allow_pickle=True)  # for multi-agent, actions.shape=(steps, num_agents)

# For simplicity, train one shared surrogate tree on first agent's actions
y = [a[0] for a in actions]  # agent 0
max_len = max(len(s) for s in states)
X = np.array([list(s) + [0]*(max_len - len(s)) for s in states])

# Train decision tree
tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X, y)

# Optional: print rules
rules = export_text(tree, feature_names=[f"P{i+1}" for i in range(max_len)])
print("Surrogate Tree Rules:\n", rules)

# Save tree for API
joblib.dump(tree, "models/multi_surrogate_tree.pkl")
print("Saved surrogate tree to models/multi_surrogate_tree.pkl")
import shap

# Create SHAP explainer
explainer = shap.TreeExplainer(tree)

# Compute SHAP values for all training states
shap_values = explainer.shap_values(X)

# Save SHAP values for API/frontend usage
np.save("logs/multi_shap_values.npy", shap_values)

print("Saved SHAP values to logs/multi_shap_values.npy")
