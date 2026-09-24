import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# =========================================================
# 1. CREATE A SIMPLE DDoS DATASET
# =========================================================

np.random.seed(42)

# Normal traffic
normal_data = {
    "packets_per_second": np.random.randint(50, 500, 500),
    "bytes_per_second": np.random.randint(5000, 50000, 500),
    "connection_count": np.random.randint(1, 30, 500),
    "packet_size": np.random.randint(200, 1500, 500),
    "failed_connections": np.random.randint(0, 5, 500),
    "label": np.zeros(500)
}

# DDoS-like abnormal traffic
ddos_data = {
    "packets_per_second": np.random.randint(1000, 10000, 500),
    "bytes_per_second": np.random.randint(100000, 1000000, 500),
    "connection_count": np.random.randint(100, 1000, 500),
    "packet_size": np.random.randint(50, 500, 500),
    "failed_connections": np.random.randint(20, 200, 500),
    "label": np.ones(500)
}


# Convert to DataFrames
normal_df = pd.DataFrame(normal_data)
ddos_df = pd.DataFrame(ddos_data)

# Combine both datasets
df = pd.concat([normal_df, ddos_df], ignore_index=True)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# =========================================================
# 2. DISPLAY DATASET
# =========================================================

print("\nDataset:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nClass Distribution:")
print(df["label"].value_counts())


# =========================================================
# 3. SEPARATE FEATURES AND TARGET
# =========================================================

X = df.drop("label", axis=1)
y = df["label"]


# =========================================================
# 4. TRAIN-TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 5. CREATE RANDOM FOREST MODEL
# =========================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# =========================================================
# 6. TRAIN MODEL
# =========================================================

model.fit(X_train, y_train)

print("\nModel training completed!")


# =========================================================
# 7. TEST MODEL
# =========================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy * 100, "%")


print("\nClassification Report:")
print(classification_report(y_test, y_pred))


print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# =========================================================
# 8. SAVE MODEL
# =========================================================

joblib.dump(model, "ddos_model.pkl")

print("\nModel saved as ddos_model.pkl")


# =========================================================
# 9. DEMONSTRATION
# =========================================================

print("\n====================================")
print("     DDoS DETECTION DEMONSTRATION")
print("====================================")


# Example 1: Normal traffic
normal_test = pd.DataFrame([{
    "packets_per_second": 200,
    "bytes_per_second": 20000,
    "connection_count": 10,
    "packet_size": 800,
    "failed_connections": 2
}])

prediction1 = model.predict(normal_test)[0]
probability1 = model.predict_proba(normal_test)[0][1] * 100


print("\nTest Case 1")
print("----------------------------")
print("Packets/sec:", 200)
print("Bytes/sec:", 20000)
print("Connections:", 10)
print("Packet size:", 800)
print("Failed connections:", 2)

if prediction1 == 0:
    print("Prediction: NORMAL TRAFFIC")
else:
    print("Prediction: DDoS ATTACK")

print("DDoS Probability:", round(probability1, 2), "%")


# Example 2: DDoS-like traffic
ddos_test = pd.DataFrame([{
    "packets_per_second": 7000,
    "bytes_per_second": 700000,
    "connection_count": 700,
    "packet_size": 100,
    "failed_connections": 150
}])

prediction2 = model.predict(ddos_test)[0]
probability2 = model.predict_proba(ddos_test)[0][1] * 100


print("\nTest Case 2")
print("----------------------------")
print("Packets/sec:", 7000)
print("Bytes/sec:", 700000)
print("Connections:", 700)
print("Packet size:", 100)
print("Failed connections:", 150)

if prediction2 == 0:
    print("Prediction: NORMAL TRAFFIC")
else:
    print("Prediction: DDoS ATTACK")

print("DDoS Probability:", round(probability2, 2), "%")


print("\n====================================")
print("          DEMONSTRATION END")
print("====================================")