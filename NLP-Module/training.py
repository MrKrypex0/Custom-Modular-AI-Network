# training.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from evaluate import load as load_metric

class TextClassifier:
    def __init__(self, model_name='distilbert-base-uncased', num_labels=2):
        # Check for CUDA availability
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
        else:
            self.device = torch.device("cpu")
            print("CUDA not available, using CPU")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    def preprocess(self, texts):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

    def train(self, train_texts, train_labels, epochs=3, batch_size=8):
        # Tokenize all texts
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        # Convert to dictionary format
        train_data = {k: v.tolist() for k, v in train_encodings.items()}
        train_data["labels"] = list(train_labels)

        # Create a Dataset object
        train_dataset = Dataset.from_dict(train_data)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            eval_strategy="no",
            logging_dir='./logs',
            logging_steps=10,
            save_strategy="epoch",
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),  # Only use fp16 if CUDA is available
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset
        )

        trainer.train()

    def evaluate(self, test_texts, test_labels):
        test_encodings = self.tokenizer(
            test_texts,
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt"
        )

        test_data = {k: v.tolist() for k, v in test_encodings.items()}
        test_data["labels"] = list(test_labels)
        test_dataset = Dataset.from_dict(test_data)

        metric = load_metric("accuracy")

        def compute_metrics(p):
            preds = p.predictions.argmax(-1)
            return metric.compute(predictions=preds, references=p.label_ids)

        training_args = TrainingArguments(
            output_dir='./results',
            per_device_eval_batch_size=8,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )

        results = trainer.evaluate()
        print(f"Accuracy: {results['eval_accuracy']}")

# Example usage:
if __name__ == "__main__":
    # Sample data for demonstration purposes
    train_texts = ["I love this product", "This is the worst experience I've had"]
    train_labels = [1, 0]  # Binary classification (positive/negative)

    test_texts = ["Absolutely fantastic!", "Terrible service."]
    test_labels = [1, 0]

    classifier = TextClassifier()
    classifier.train(train_texts, train_labels)
    classifier.evaluate(test_texts, test_labels)
