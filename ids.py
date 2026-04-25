import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# NSL-KDD columns
with open("feature_names.txt") as f:
    cols = [line.strip() for line in f]

# Load data
train = pd.read_csv("KDDTrain+.txt", names=cols)
test = pd.read_csv("KDDTest+.txt", names=cols)

# Verify dataset dimensions
assert train.shape[1] == 43
assert test.shape[1] == 43

print("Training set shape:", train.shape)
print("Test set shape:", test.shape)


# Convert attack labels to binary classification:
# normal traffic -> 0
# attack traffic -> 1
train['label'] = train['label'].apply(
    lambda x: 0 if str(x).strip().startswith("normal") else 1
)

test['label'] = test['label'].apply(
    lambda x: 0 if str(x).strip().startswith("normal") else 1
)


# Label-encode categorical features
for col in ['protocol_type', 'service', 'flag']:

    le = LabelEncoder()
    allvals = pd.concat([train[col], test[col]])
    le.fit(allvals)

    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])


# Split features (X) and target labels (y)
X_train = train.drop('label', axis=1)
y_train = train['label']

X_test = test.drop('label', axis=1)
y_test = test['label']


# Replace missing values
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)


# ================= RANDOM FOREST =================

model_rf = RandomForestClassifier(
    n_estimators=500,
    class_weight='balanced',
    max_depth=None,
    random_state=42
)

model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)

print("\n===== Random Forest Results =====")
print(classification_report(y_test, pred_rf))
print("Test Accuracy:", model_rf.score(X_test, y_test))

cm_rf = confusion_matrix(y_test, pred_rf)
print("Confusion Matrix:")
print(cm_rf)


# ================= LOGISTIC REGRESSION =================

# Scale features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_lr = LogisticRegression(
    class_weight='balanced',
    max_iter=5000,
    random_state=42
)

model_lr.fit(X_train_scaled, y_train)
pred_lr = model_lr.predict(X_test_scaled)

print("\n===== Logistic Regression Results =====")
print(classification_report(y_test, pred_lr))
print("Test Accuracy:", model_lr.score(X_test_scaled, y_test))

cm_lr = confusion_matrix(y_test, pred_lr)
print("Confusion Matrix:")
print(cm_lr)

plt.figure(figsize=(12,5))

# Random Forest
plt.subplot(1,2,1)
plt.imshow(cm_rf, interpolation='nearest')
plt.title("Random Forest")
plt.colorbar()
plt.xticks([0,1], ['Normal','Attack'])
plt.yticks([0,1], ['Normal','Attack'])
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm_rf[i,j], ha='center', va='center')


# Logistic Regression
plt.subplot(1,2,2)
plt.imshow(cm_lr, interpolation='nearest')
plt.title("Logistic Regression")
plt.colorbar()
plt.xticks([0,1], ['Normal','Attack'])
plt.yticks([0,1], ['Normal','Attack'])
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm_lr[i,j], ha='center', va='center')


plt.tight_layout()
plt.show()