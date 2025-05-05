# model.py
from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

# Simple data: age → 0 (child) or 1 (adult)
X = np.array([[5], [12], [17], [18], [25], [35]])
y = np.array([0, 0, 0, 1, 1, 1])

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model to file
joblib.dump(model, "age_model.pkl")
