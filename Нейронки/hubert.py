import numpy as np
import evaluate
from datasets import load_dataset, Audio
from transformers import (
    AutoFeatureExtractor, 
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer
)


max_duration = 30.0

metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids, average='macro')

def preprocess_function(examples, feature_extractor):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
        return_attention_mask=True,
    )
    return inputs


def main():
    gtzan = load_dataset("marsyas/gtzan", "all")
    gtzan = gtzan["train"].train_test_split(seed=42, shuffle=True, test_size=0.1)
    id2label_fn = gtzan["train"].features["genre"].int2str
    model_id = "ntu-spml/distilhubert"
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_id, do_normalize=True, return_attention_mask=True
    )
    sampling_rate = feature_extractor.sampling_rate
    gtzan = gtzan.cast_column("audio", Audio(sampling_rate=sampling_rate))
    gtzan_encoded = gtzan.map(
        preprocess_function,
        remove_columns=["audio", "file"],
        batched=True,
        batch_size=100,
        num_proc=1,
    )
    gtzan_encoded = gtzan_encoded.rename_column("genre", "label")
    id2label = {
        str(i): id2label_fn(i)
        for i in range(len(gtzan_encoded["train"].features["label"].names))
    }
    label2id = {v: k for k, v in id2label.items()}

    num_labels = len(id2label)

    model = AutoModelForAudioClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    model_name = model_id.split("/")[-1]
    batch_size = 8
    gradient_accumulation_steps = 1
    num_train_epochs = 10

    training_args = TrainingArguments(
        f"{model_name}-finetuned-gtzan",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.1,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=gtzan_encoded["train"],
        eval_dataset=gtzan_encoded["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == '__main__':
    main()