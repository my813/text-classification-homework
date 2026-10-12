import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42

# 读取数据
df = pd.read_csv("data/nyt.csv")
df = df.dropna(subset=["text", "label"]).reset_index(drop=True)

# 先分80%训练，20%临时
try:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=temp_df["label"]
    )
except ValueError:
    # 如果某些类别样本太少，stratify报错，就退回普通随机划分
    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE
    )

# 保存划分结果
train_df.to_csv("data/nyt_train.csv", index=False)
val_df.to_csv("data/nyt_val.csv", index=False)
test_df.to_csv("data/nyt_test.csv", index=False)

print("训练集：", len(train_df))
print("验证集：", len(val_df))
print("测试集：", len(test_df))