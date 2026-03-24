import pandas as pd
import numpy as np

# 資料不平衡，未處理

# =========================
# 1. 讀資料
# =========================
train = pd.read_csv(r"C:\Users\aa090\OneDrive\桌面\員工離職_AI訓練\data\train.csv")
test = pd.read_csv(r"C:\Users\aa090\OneDrive\桌面\員工離職_AI訓練\data\test.csv")
season = pd.read_csv(r"C:\Users\aa090\OneDrive\桌面\員工離職_AI訓練\data\season.csv")

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

# =========================
# 5. 歷史特徵
# =========================
lag_cols = [
    "專案時數", "專案總數", "訓練時數A", "訓練時數B", "訓練時數C",
    "生產總額", "榮譽數", "升遷速度",
    "近三月請假數A", "近一年請假數A",
    "近三月請假數B", "近一年請假數B",
    "出差數A", "出差數B",
    "年度績效等級A", "年度績效等級B", "年度績效等級C",
    "加班數_sum", "加班數_mean",
    "請假數A_sum", "請假數B_sum",
    "出差數A_sum", "出差數B_sum"
]

for col in lag_cols:
    if col in all_df.columns:
        prev = all_df.groupby("PerNo")[col].shift(1)
        all_df[f"{col}_lag1"] = prev
        all_df[f"{col}_diff1"] = all_df[col] - prev

# =========================
# 6. rolling / 累積特徵
# =========================
rolling_cols = ["加班數_sum", "請假數A_sum", "請假數B_sum", "出差數A_sum", "出差數B_sum"]

for col in rolling_cols:
    if col in all_df.columns:
        all_df[f"{col}_rolling2_mean"] = (
            all_df.groupby("PerNo")[col]
            .transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
        )
        all_df[f"{col}_cumsum_prev"] = (
            all_df.groupby("PerNo")[col]
            .transform(lambda s: s.shift(1).cumsum())
        )

# =========================
# 7. 比例特徵
# =========================
if "加班數_sum" in all_df.columns and "請假數A_sum" in all_df.columns:
    all_df["加班請假比_A"] = all_df["加班數_sum"] / (all_df["請假數A_sum"] + 1)

if "加班數_sum" in all_df.columns and "請假數B_sum" in all_df.columns:
    all_df["加班請假比_B"] = all_df["加班數_sum"] / (all_df["請假數B_sum"] + 1)

if "榮譽數" in all_df.columns and "專案總數" in all_df.columns:
    all_df["每專案榮譽數"] = all_df["榮譽數"] / (all_df["專案總數"] + 1)

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