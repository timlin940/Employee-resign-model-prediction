import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, fbeta_score
from sklearn.metrics import precision_score, recall_score
from pathlib import Path

# 根據題目 recall 比 precision 更重要，所以選擇 F-beta, beta=1.5 作為評分指標

script_dir = Path(__file__).parent
project_dir = script_dir.parent

# 讀資料
X = pd.read_csv(project_dir / "asset" / "output_data" / "X_processed.csv")
y = pd.read_csv(project_dir / "asset" / "output_data" / "y_processed.csv")["PerStatus"]

X_test = pd.read_csv(project_dir / "asset" / "output_data" / "X_test_processed.csv")
test_id = pd.read_csv(project_dir / "asset" / "output_data" / "test_id.csv")
# 切 train / valid
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 模型
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.01,
    num_leaves=32,
    random_state=42,
    class_weight={0:1, 1:15} #資料不平均，調整權重
)

model.fit(X_train, y_train)

# validation 機率
pred_valid_proba = model.predict_proba(X_valid)[:, 1]

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

pred_label = (pred_valid_proba >= best_th).astype(int)

print("Precision:", precision_score(y_valid, pred_label))
print("Recall:", recall_score(y_valid, pred_label))
print("Accuracy:", (pred_label == y_valid).mean())

submission.to_csv("output_data/answer.csv", index=False, encoding="utf-8-sig")
print("answer.csv 已輸出")


# 過濾噪音（這個會直接提升 precision）

# 你現在所有特徵都丟進去，其實會：

# 增加 FP（降低 precision）

# 做法：看 feature importance
import pandas as pd

importance = model.feature_importances_
feat_imp = pd.Series(importance, index=X.columns).sort_values(ascending=False)
feat_imp.to_csv("output_data/feature_importance.csv", encoding="utf-8-sig")