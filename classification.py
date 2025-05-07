import glob
import json
import os
import torch
import pandas as pd
import transformers
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLLModel:
    def __init__(self, model_name, pipeline=None):
        self.model_name = model_name
        if pipeline is not None:
            self.model = pipeline
        else:
            self.model = transformers.pipeline(
                "text-generation",
                model=model_name,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )

    def generate_response(self, prompt):
        print(f"Generating response using {self.model_name}...")
        messages = [
            {"role": "system", "content": "You are an expert document classifier."},
            {"role": "user", "content": prompt},
        ]

        outputs = self.model(
            messages,
            max_new_tokens=4096,
        )

        return outputs[0]["generated_text"][-1]["content"].replace("\n", "")


def load_lora_model(base_model_name, lora_path):

    print(f"Loading base model: {base_model_name} with LoRA from: {lora_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Apply the LoRA weights
    model = PeftModel.from_pretrained(base_model, lora_path)

    # Important: model must be in eval mode for inference
    model.eval()

    # Create a generation pipeline manually
    pipe = transformers.TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )

    return pipe


# Preprocess dataset
def preprocess_dataset(data_folder, max_files=None):
    print("Preprocessing dataset...")

    # Cargar el diccionario de etiquetas desde eurovoc_en.json
    eurovoc_path = os.path.join(data_folder, "../eurovoc_en.json")
    with open(eurovoc_path, "r", encoding="utf-8") as f:
        eurovoc_data = json.load(f)

    # Crear un diccionario {id: label}
    eurovoc_labels = {
        item["concept_id"]: item["label"] for item in eurovoc_data.values()
    }

    # Obtener archivos JSON del dataset
    json_files = glob.glob(f"{data_folder}/*.json")
    if max_files:
        json_files = json_files[:max_files]

    records = []

    for file in json_files:
        if os.path.basename(file) == "eurovoc_en.json":
            continue  # Omitir el archivo de etiquetas

        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Unir contenido textual
        text_parts = [
            data.get("title", ""),
            data.get("header", ""),
            data.get("recitals", ""),
            " ".join(data.get("main_body", [])),
        ]
        full_text = "\n".join([part for part in text_parts if part])

        # Convertir los conceptos a etiquetas de texto
        concept_ids = data.get("concepts", [])
        concept_labels = [eurovoc_labels.get(cid, "") for cid in concept_ids]
        concept_labels = [label for label in concept_labels if label]  # eliminar vacíos
        labels_str = ", ".join(concept_labels)

        records.append(
            {
                "text": full_text,
                "concepts": labels_str,
            }
        )

    # Retornar el DataFrame y la lista de posibles etiquetas
    return pd.DataFrame(records), list(eurovoc_labels.values())


embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def evaluate_semantic_similarity(y_true, y_pred, threshold=0.7):
    y_true_cleaned = str(y_true).lower().strip()
    y_pred_cleaned = str(y_pred).lower().strip()

    # Generar embeddings
    true_embs = embedder.encode(y_true_cleaned, convert_to_tensor=True)
    pred_embs = embedder.encode(y_pred_cleaned, convert_to_tensor=True)

    # Asegurar que sean 2D
    true_embs = true_embs.unsqueeze(0).cpu().numpy()
    pred_embs = pred_embs.unsqueeze(0).cpu().numpy()

    # Calcular similitud
    sim = cosine_similarity(true_embs, pred_embs)[0][0]

    return sim


# Evaluate model output against ground truth
def evaluate_model_output(y_pred, y_true):
    # Calculate BLEU score with smoothing
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu(
        [y_true.split()],
        y_pred.split(),
        smoothing_function=smoothing_function,
    )

    # Calculate ROUGE scores
    rouge = Rouge()
    rouge_scores = rouge.get_scores(y_pred, y_true, avg=True)

    # Evaluate semantic similarity
    semantic_validity = evaluate_semantic_similarity(y_true, y_pred)

    # Convertir a sets para evaluar la clasificación
    pred_set = set(y_pred.strip().split())
    true_set = set(map(str, y_true))

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    exact_match = pred_set == true_set

    return {
        "bleu": bleu_score,
        "rouge": rouge_scores,
        "semantic_validity": semantic_validity,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
    }


# Main function
def main():
    data_folder = "datasets/classification/train"

    lora_pipeline = load_lora_model(
        "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "./lora/llama3-finetuned",
    )

    models = [
        LLLModel("NousResearch/Meta-Llama-3.1-8B-Instruct"),
        LLLModel("Qwen/Qwen2.5-7B-Instruct"),
        LLLModel(
            "NousResearch/Meta-Llama-3.1-8B-Instruct + LoRA", pipeline=lora_pipeline
        ),
    ]

    # Load dataset
    df, _ = preprocess_dataset(data_folder)
    df = df[:10]

    for model in models:
        model_name = model.model_name
        metrics = []
        for _, row in df.iterrows():

            prompt = f"""
            You are an expert document analyzer. Your task is to classify the document content provided below. Respond with labels separated by spaces. use as much words as u can. Classify specific topics about the contents of the document

            Instructions:
            - Do not include any explanation.
            - respond with labels only.

            Document content:
            \"\"\"{df["text"]}\"\"\""""
            model_output = model.generate_response(prompt)
            print(f"Model output: {model_output}")
            print(f"Concepts: {row['concepts']}")

            # Evaluate the model output
            evaluation = evaluate_model_output(model_output, row["concepts"])
            metrics.append(evaluation)

        # Calculate average metrics for the model
        avg_bleu = sum(m["bleu"] for m in metrics) / len(metrics)
        avg_rouge = {
            key: sum(m["rouge"][key]["f"] for m in metrics) / len(metrics)
            for key in metrics[0]["rouge"].keys()
        }
        semantic_accuracy = sum([m["semantic_validity"] for m in metrics]) / len(
            metrics
        )
        avg_precision = sum([m["precision"] for m in metrics]) / len(metrics)
        avg_recall = sum([m["recall"] for m in metrics]) / len(metrics)
        avg_f1 = sum([m["f1"] for m in metrics]) / len(metrics)
        avg_exact_match = sum([m["exact_match"] for m in metrics]) / len(metrics)

        print(
            f"Model: {model_name}, Average BLEU: {avg_bleu}, Average ROUGE: {avg_rouge}, Semantic Accuracy: {semantic_accuracy}, Average Precision: {avg_precision}, Average Recall: {avg_recall}, Average F1: {avg_f1}, Average Exact Match: {avg_exact_match}"
        )


if __name__ == "__main__":
    main()
