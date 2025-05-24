# 1. وارد کردن کتابخونه‌ها
import pandas as pd
import csv
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

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
# 7. تنظیمات فاین‌تیون
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
    save_total_limit=2,
    fp16=True
)

# 8. تعریف متریک
def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")  # استفاده از evaluate.load
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# 9. تعریف Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)# 10. اجرای تست
test_results = trainer.evaluate()
with open("test_results.txt", "w") as f:
    f.write(f"Test Accuracy: {test_results['eval_accuracy']}\n")