import os
import inspect
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

# 关键：设置 Hugging Face 国内镜像，避免下载模型时出现 SSL 报错
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

RANDOM_STATE = 42
MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LEN = 64
EPOCHS = 3

# 读取数据
train = pd.read_csv("data/nyt_train.csv")
val = pd.read_csv("data/nyt_val.csv")
test = pd.read_csv("data/nyt_test.csv")

# 标签编码（BERT 需要数字标签）
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train["label"].astype(str))
y_val = label_encoder.transform(val["label"].astype(str))
y_test = label_encoder.transform(test["label"].astype(str))
num_labels = len(label_encoder.classes_)

print(f"共有 {num_labels} 个类别，正在加载 BERT 模型...")

# 加载 BERT
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels
)

class NewsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = NewsDataset(train["text"], y_train)
val_dataset = NewsDataset(val["text"], y_val)
test_dataset = NewsDataset(test["text"], y_test)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro")
    }

# 兼容不同版本的 transformers
sig = inspect.signature(TrainingArguments.__init__)
eval_key = "evaluation_strategy" if "evaluation_strategy" in sig.parameters else "eval_strategy"

args_kwargs = dict(
    output_dir="./bert_nyt",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=50,
    report_to="none",
    seed=RANDOM_STATE,
    fp16=torch.cuda.is_available(), # 有 GPU 自动开启混合精度加速
    save_total_limit=1,
)
args_kwargs[eval_key] = "epoch"

training_args = TrainingArguments(**args_kwargs)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("\n开始微调 BERT（3 epochs）...")
trainer.train()

print("\n在测试集上进行预测...")
pred_output = trainer.predict(test_dataset)
preds = np.argmax(pred_output.predictions, axis=-1)

acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="macro")

print("\n" + "="*40)
print("BERT-base-uncased 最终结果")
print(f"Accuracy = {acc:.4f}")
print(f"Macro-F1 = {f1:.4f}")
print("="*40)