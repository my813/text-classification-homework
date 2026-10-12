import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# 自定义简单的分词器：转小写，用正则提取纯英文字母单词
def tokenize(text):
    return re.findall(r'\b[a-z]+\b', str(text).lower())

# 读取统一划分后的数据
train = pd.read_csv("data/nyt_train.csv")
test = pd.read_csv("data/nyt_test.csv")

X_train_text = train["text"].astype(str)
y_train = train["label"]
X_test_text = test["text"].astype(str)
y_test = test["label"]

def run_bow(name, binary):
    # binary=True  -> Binary Bag of Words
    # binary=False -> Word Frequency
    vectorizer = CountVectorizer(
        tokenizer=tokenize,
        lowercase=False,
        token_pattern=None, # 显式设为 None，消除 sklearn 的警告
        binary=binary
    )
    # 词表只从训练集学习
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")

    print(f"\n{name}")
    print(f"Accuracy = {acc:.4f}")
    print(f"Macro-F1 = {f1:.4f}")
    print("-" * 40)

# 1. Binary Bag of Words
run_bow("Binary Bag of Words + Logistic Regression", binary=True)

# 2. Word Frequency
run_bow("Word Frequency + Logistic Regression", binary=False)