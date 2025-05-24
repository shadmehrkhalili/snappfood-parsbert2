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