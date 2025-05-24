# 1. وارد کردن کتابخونه‌ها
import pandas as pd
import csv
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 2. لود کردن دیتاست‌ها
train_df = pd.read_csv("train.csv", sep="\t", quoting=csv.QUOTE_ALL, encoding="utf-8")
dev_df = pd.read_csv("dev.csv", sep="\t", quoting=csv.QUOTE_ALL, encoding="utf-8")
test_df = pd.read_csv("test.csv", sep="\t", quoting=csv.QUOTE_ALL, encoding="utf-8")

# حذف ستون‌های اضافی
train_df = train_df[["comment", "label_id"]]
dev_df = dev_df[["comment", "label_id"]]
test_df = test_df[["comment", "label_id"]]

# تبدیل به فرمت Dataset
train_dataset = Dataset.from_pandas(train_df)
dev_dataset = Dataset.from_pandas(dev_df)
test_dataset = Dataset.from_pandas(test_df)

datasets = DatasetDict({
    "train": train_dataset,
    "validation": dev_dataset,
    "test": test_dataset
})
# 3. لود کردن توکنایزر
tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")

# 4. توکنایز کردن داده‌ها
def tokenize_function(examples):
    return tokenizer(examples["comment"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = datasets.map(tokenize_function, batched=True)
# 5. آماده‌سازی داده‌ها
tokenized_datasets = tokenized_datasets.remove_columns(["comment"])
tokenized_datasets = tokenized_datasets.rename_column("label_id", "labels")
tokenized_datasets.set_format("torch")

# 6. لود کردن مدل
model = AutoModelForSequenceClassification.from_pretrained("HooshvareLab/bert-fa-base-uncased", num_labels=2)