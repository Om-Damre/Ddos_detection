import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

np.random.seed(42)

# ==========================================
# 1. CREATE MORE REALISTIC SYNTHETIC DATA
# ==========================================

n = 2000

# --------------------------
# NORMAL TRAFFIC
# --------------------------

normal = pd.DataFrame({
    "packets_per_second": np.random.normal(500, 300, n),
    "bytes_per_second": np.random.normal(50000, 30000, n),
    "connection_count": np.random.normal(50, 30, n),
    "packet_size": np.random.normal(700, 200, n),
    "failed_connections": np.random.normal(5, 4, n)
})

normal["label"] = 0


# --------------------------
# DDOS TRAFFIC
# --------------------------

ddos = pd.DataFrame({
    "packets_per_second": np.random.normal(2500, 1200, n),
    "bytes_per_second": np.random.normal(250000, 120000, n),
    "connection_count": np.random.normal(300, 150, n),
    "packet_size": np.random.normal(350, 150, n),
    "failed_connections": np.random.normal(60, 30, n)
})

ddos["label"] = 1


# ==========================================
# 2. COMBINE DATA
# ==========================================

df = pd.concat([normal, ddos], ignore_index=True)

# Remove negative values
feature_columns = [
    "packets_per_second",
    "bytes_per_second",
    "connection_count",
    "packet_size",
    "failed_connections"
]

for column in feature_columns:
    df[column] = df[column].clip(lower=0)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# ==========================================
# 3. FEATURES AND TARGET
# ==========================================

X = df[feature_columns]
y = df["label"]


# ==========================================
# 4. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==========================================
# 5. RANDOM FOREST
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    min_samples_leaf=10,
    random_state=42
)


# ==========================================
# 6. TRAIN
# ==========================================

model.fit(X_train, y_train)


# ==========================================
# 7. EVALUATION
# ==========================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("MODEL RESULTS")
print("================================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==========================================
# 8. SAVE MODEL
# ==========================================

joblib.dump(model, "ddos_model.pkl")

print("\nModel saved successfully!")
print("File: ddos_model.pkl")