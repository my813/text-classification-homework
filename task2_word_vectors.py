import os
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from gensim.models import Word2Vec

# 与 Task 1 一致的分词器
def tokenize(text):
    return re.findall(r'\b[a-z]+\b', str(text).lower())

# 读取统一划分后的数据
train = pd.read_csv("data/nyt_train.csv")
test = pd.read_csv("data/nyt_test.csv")

y_train = train["label"]
y_test = test["label"]

print("正在对文本进行分词...")
train_tokens = [tokenize(t) for t in train["text"]]
test_tokens = [tokenize(t) for t in test["text"]]

# 文档向量平均函数
def average_vectors(tokens_list, kv, dim=100):
    X = np.zeros((len(tokens_list), dim), dtype=np.float32)
    for i, tokens in enumerate(tokens_list):
        vecs = []
        for word in tokens:
            if word in kv:
                vecs.append(kv[word])
        if len(vecs) > 0:
            X[i] = np.mean(vecs, axis=0)
    return X

# 统一的评估函数
def evaluate(name, X_train, X_test):
    print(f"\n正在训练和评估: {name} ...")
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    print(f"{name}")
    print(f"Accuracy = {acc:.4f}")
    print(f"Macro-F1 = {f1:.4f}")
    print("-" * 40)

# 1. 预训练 GloVe 100 维
glove_path = "data/glove.6B.100d.txt"

if os.path.exists(glove_path):
    print("\n正在加载 GloVe 100d 词向量，请稍等...")
    glove = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != 101: # 1个单词 + 100个数字
                continue
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            glove[word] = vec

    print("GloVe 加载完成，正在计算文档向量...")
    X_train = average_vectors(train_tokens, glove)
    X_test = average_vectors(test_tokens, glove)

    evaluate("GloVe 100d average + Logistic Regression", X_train, X_test)
else:
    print(f"错误：未找到 {glove_path}，跳过 GloVe。")

# 2. 用 AG News 训练 Word2Vec
ag_path = "data/ag.csv"
if os.path.exists(ag_path):
    print("\n正在用 AG News 训练 Word2Vec ...")
    ag_df = pd.read_csv(ag_path)
    
    # 提取 AG News 文本
    if "text" in ag_df.columns:
        ag_text = ag_df["text"].astype(str)
    else:
        ag_text = ag_df.iloc[:, 0].astype(str)
        
    ag_tokens = [tokenize(t) for t in ag_text]

    w2v_ag = Word2Vec(
        sentences=ag_tokens,
        vector_size=100,
        window=5,
        min_count=1,
        sg=1,
        workers=4,
        epochs=5,
        seed=42
    )
    
    print("AG News Word2Vec 训练完成，正在计算文档向量...")
    X_train = average_vectors(train_tokens, w2v_ag.wv)
    X_test = average_vectors(test_tokens, w2v_ag.wv)

    evaluate("Word2Vec AG News 100d average + Logistic Regression", X_train, X_test)
else:
    print(f"错误：未找到 {ag_path}，跳过 AG News Word2Vec。")

# 3. 用 NYT 训练集训练 Word2Vec
print("\n正在用 NYT 训练集训练 Word2Vec ...")
w2v_nyt = Word2Vec(
    sentences=train_tokens,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    workers=4,
    epochs=5,
    seed=42
)

print("NYT Word2Vec 训练完成，正在计算文档向量...")
X_train = average_vectors(train_tokens, w2v_nyt.wv)
X_test = average_vectors(test_tokens, w2v_nyt.wv)

evaluate("Word2Vec NYT 100d average + Logistic Regression", X_train, X_test)