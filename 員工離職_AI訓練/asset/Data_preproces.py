import pandas as pd
import numpy as np
from pathlib import Path
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

# 需要特別做計算的是一個人season的工作表現計算(加班、出差、請假)

# =========================
# 5. 建立前一年變化特徵
# =========================
all_df = all_df.sort_values(["PerNo", "yyyy"]).reset_index(drop=True)

# 你想追蹤變化的欄位
change_cols = [
    "生產總額",
    "年度績效等級A",  "年度績效等級B","年度績效等級C",
    "加班數_sum", "加班數_mean", "加班數_max",
    "榮譽數",
    "升遷速度"
]

# 只對存在的欄位做，避免報錯
change_cols = [col for col in change_cols if col in all_df.columns]

for col in change_cols:
    # 前一年數值
    all_df[f"{col}_prev1"] = all_df.groupby("PerNo")[col].shift(1)

    # 與前一年差值
    all_df[f"{col}_diff1"] = all_df[col] - all_df[f"{col}_prev1"]

# =========================
# 8. 缺失值處理
# 只用 train 的統計量補值，避免資料洩漏
# =========================

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
drop_cols = ["PerNo", "yyyy", "is_train", target,
            "請假數B_mean",
            "出差數B_mean",
            "出差數A_mean",
            "加班數_mean",
            "請假數A_mean",
            "加班數_mean_prev1",
            "加班數_mean_diff1"
    ]

# ===================
# 暴力審查資料:手動篩選我個人認為敏感的數據
# ===================

X = train_part.drop(columns=drop_cols, errors="ignore")
y = train_part[target]
X_test = test_part.drop(columns=drop_cols, errors="ignore")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("X_test shape:", X_test.shape)

# 儲存處理後資料
X.to_csv("output_data/X_processed.csv", index=False)
y.to_csv("output_data/y_processed.csv", index=False)
X_test.to_csv("output_data/X_test_processed.csv", index=False)

# 如果 submission 需要員工編號，可另外存
test_part[["PerNo", "yyyy"]].to_csv("output_data/test_id.csv", index=False)