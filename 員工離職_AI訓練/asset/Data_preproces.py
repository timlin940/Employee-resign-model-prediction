import pandas as pd
import numpy as np
from pathlib import Path

# 資料問題
# 1. 不平衡，太多未離職，需要再資料或是模型上去做加權
# 2. 資料沒有正規化、標準化

# =========================
# 1. 讀資料
# =========================

script_dir = Path(__file__).parent
project_dir = script_dir.parent

train = pd.read_csv(project_dir / "data" / "train.csv")
test = pd.read_csv(project_dir / "data" / "test.csv")
season = pd.read_csv(project_dir / "data" / "season.csv")

# 標記資料來源
train["is_train"] = 1
test["is_train"] = 0

# test 沒有標籤，先補欄位方便 concat
if "PerStatus" not in test.columns:
    test["PerStatus"] = np.nan

# 合併 train + test
all_df = pd.concat([train, test], axis=0, ignore_index=True)

# =========================
# 2. season: 季資料 -> 年資料
# =========================
season_year = (
    season.groupby(["PerNo", "yyyy"], as_index=False)
    .agg({
        "加班數": ["sum", "mean", "max"],
        "出差數A": ["sum", "mean"],
        "出差數B": ["sum", "mean"],
        "請假數A": ["sum", "mean"],
        "請假數B": ["sum", "mean"],
    })
)

season_year.columns = [
    "PerNo", "yyyy",
    "加班數_sum", "加班數_mean", "加班數_max",
    "出差數A_sum", "出差數A_mean",
    "出差數B_sum", "出差數B_mean",
    "請假數A_sum", "請假數A_mean",
    "請假數B_sum", "請假數B_mean"
]

# =========================
# 3. 合併主表 + season_year
# =========================
all_df = all_df.merge(season_year, on=["PerNo", "yyyy"], how="left")

# =========================
# 4. 排序
# =========================
all_df = all_df.sort_values(["PerNo", "yyyy"]).reset_index(drop=True)

# 需要特別做計算的是[]

# =========================
# 8. 缺失值處理
# 只用 train 的統計量補值，避免資料洩漏
# =========================
# 刪掉不必要的欄位
all_df.drop(columns=["歸屬部門","廠區代碼","工作地點","當前專案角色"], inplace=True)

train_part = all_df[all_df["is_train"] == 1].copy()
test_part = all_df[all_df["is_train"] == 0].copy()

num_cols = all_df.select_dtypes(include=[np.number]).columns.tolist()

# 不要補 target
if "PerStatus" in num_cols:
    num_cols.remove("PerStatus")

fill_values = {}
for col in num_cols:
    fill_values[col] = train_part[col].median()

train_part[num_cols] = train_part[num_cols].fillna(fill_values)
test_part[num_cols] = test_part[num_cols].fillna(fill_values)

# =========================
# 9. 分出 X / y / X_test
# =========================
target = "PerStatus"
drop_cols = ["PerNo", "yyyy", "is_train", target]

X = train_part.drop(columns=drop_cols, errors="ignore")
y = train_part[target]
X_test = test_part.drop(columns=drop_cols, errors="ignore")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X_test shape:", X_test.shape)

# 儲存處理後資料
X.to_csv("X_processed.csv", index=False)
y.to_csv("y_processed.csv", index=False)
X_test.to_csv("X_test_processed.csv", index=False)

# 如果 submission 需要員工編號，可另外存
test_part[["PerNo", "yyyy"]].to_csv("test_id.csv", index=False)