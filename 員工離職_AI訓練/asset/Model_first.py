import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, fbeta_score

# 讀資料
X = pd.read_csv(r"C:\Users\aa090\OneDrive\桌面\員工離職_AI訓練\asset\X_processed.csv")
y = pd.read_csv(r"C:\Users\aa090\OneDrive\桌面\員工離職_AI訓練\asset\y_processed.csv").squeeze()

X_test = pd.read_csv(r"C:\Users\aa090\OneDrive\桌面\員工離職_AI訓練\asset\X_test_processed.csv")
test_id = pd.read_csv(r"C:\Users\aa090\OneDrive\桌面\員工離職_AI訓練\asset\test_id.csv")

# 切 train / valid
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 模型
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# validation 機率
pred_valid_proba = model.predict_proba(X_valid)[:, 1]

# 先看 AUC（可當參考）
auc = roc_auc_score(y_valid, pred_valid_proba)
print("AUC:", auc)

# 找最佳 threshold（針對 F-beta, beta=1.5）
best_th = 0.5
best_score = -1

for th in np.arange(0.05, 0.96, 0.01):
    pred_valid_label = (pred_valid_proba >= th).astype(int)
    score = fbeta_score(y_valid, pred_valid_label, beta=1.5)
    
    if score > best_score:
        best_score = score
        best_th = th

print("Best threshold:", round(best_th, 2))
print("Best F1.5:", best_score)

# 用最佳 threshold 預測 test
pred_test_proba = model.predict_proba(X_test)[:, 1]
pred_test_label = (pred_test_proba >= best_th).astype(int)

submission = pd.DataFrame({
    "PerNo": test_id["PerNo"],
    "PerStatus": pred_test_label
})

submission.to_csv("submission.csv", index=False, encoding="utf-8-sig")
print("submission.csv 已輸出")