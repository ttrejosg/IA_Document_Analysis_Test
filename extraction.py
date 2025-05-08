import io
import cv2
import glob
import json
import torch
import numpy as np
import pandas as pd
import transformers
from PIL import Image
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pytesseract
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class LLLModel:
    def __init__(self, model_name, cache_dir="model_cache"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        self.model = pipeline

    def generate_response(self, prompt):
        print(f"Generating response using {self.model_name}...")
        messages = [
            {"role": "system", "content": "You are an expert document analyzer."},
            {"role": "user", "content": prompt},
        ]

        outputs = self.model(
            messages,
            max_new_tokens=4096,
        )

        return (
            outputs[0]["generated_text"][-1]["content"]
            .replace("json", "")
            .replace("```", "")
            .replace("\n", "")
        )


# Preprocess image for better OCR results
def preprocess_image(image_data):
    try:
        # Convert binary data to a NumPy array
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to improve contrast
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        # Denoise the image
        denoised = cv2.fastNlMeansDenoising(binary, None, 30, 7, 21)

        return denoised
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


# Extract text from image using EasyOCR
def extract_text_from_image(image_data):
    try:
        # Preprocess the image
        preprocessed_image = preprocess_image(image_data)
        if preprocessed_image is None:
            return ""

        # Use EasyOCR to extract text
        # reader = easyocr.Reader(["en"], gpu=True)
        # results = reader.readtext(preprocessed_image, detail=0)

        # Combine results into a single string
        # return " ".join(results).strip()
        #
        # Use pytesseract to extract text
        try:
            text = pytesseract.image_to_string(preprocessed_image, lang="eng")
            # Remove any leading/trailing whitespace
            return text.strip()
        except Exception as e:
            print(f"Error extracting text from image: {e}")
            return ""
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return ""


def show_image(image_data):
    try:
        # Convert binary data to a BytesIO object
        image = Image.open(io.BytesIO(image_data))
        image.show()  # This will open the image in the default image viewer
        image.save("output.png")
    except Exception as e:
        print(f"Error displaying image: {e}")


# Preprocess dataset
def preprocess_dataset(data_folder, max_files=None):
    print("Preprocessing dataset...")
    # Read all .parquet files in the folder
    parquet_files = glob.glob(f"{data_folder}/*.parquet")
    print(parquet_files)

    # Limit the number of files to read if max_files is specified
    if max_files is not None:
        parquet_files = parquet_files[:max_files]

    # Combine all selected parquet files into a single DataFrame
    df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
    df = df[:2]
    show_image(df["image"][0]["bytes"])

    # Convert 'image' column to text (placeholder logic for OCR)
    df["image_text"] = df["image"].apply(lambda x: extract_text_from_image(x["bytes"]))
    print(df["image_text"][0])

    # Obtener el 'gt_parse', que es la respuesta esperada
    def clean_ground_truth(gt):
        gt_dict = json.loads(gt) if isinstance(gt, str) else gt
        return gt_dict.get("gt_parse", gt_dict)

    df["ground_truth"] = df["ground_truth"].apply(clean_ground_truth)
    return df


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
def evaluate_model_output(model_output, ground_truth):
    # Convert ground_truth to a JSON string if it's a dictionary
    if isinstance(ground_truth, dict):
        ground_truth = json.dumps(ground_truth)

    # Ensure model_output is a string
    if isinstance(model_output, dict):
        model_output = json.dumps(model_output)

    # Calculate BLEU score with smoothing
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu(
        [ground_truth.split()],
        model_output.split(),
        smoothing_function=smoothing_function,
    )

    # Calculate ROUGE scores
    rouge = Rouge()
    rouge_scores = rouge.get_scores(model_output, ground_truth, avg=True)
    semantic_validity = evaluate_semantic_similarity(ground_truth, model_output)

    return {
        "bleu": bleu_score,
        "rouge": rouge_scores,
        "semantic_validity": semantic_validity,
    }


# Main function
def main():
    data_folder = "datasets/extraction"
    models = [
        LLLModel("NousResearch/Meta-Llama-3-8B-Instruct"),
        LLLModel("Qwen/Qwen2.5-7B-Instruct"),
    ]

    # Load dataset
    df = preprocess_dataset(data_folder)

    for model in models:
        model_name = model.model_name
        metrics = []
        for _, row in df.iterrows():
            prompt = f"""You are an expert document analyzer. Your task is to extract the key information from the invoice.

            This is the list of posible fields:
                [
                menu, menu.nm, menu.num, menu.unitprice, menu.cnt, menu.discountprice, menu.price, menu.itemsubtotal, menu.vatyn, menu.etc, menu.sub_nm, menu.sub_num, menu.sub_unitprice, menu.sub_cnt, menu.sub_discountprice, menu.sub_price, menu.sub_etc, void_menu, void_menu.nm, voidmenu.num, voidmenu.unitprice, voidmenu.cnt, void_menu.price, voidmenu.etc, subtotal, subtotal.subtotal_price, subtotal.discount_price, subtotal.subtotal_count, subtotal.service_price, subtotal.othersvc_price, subtotal.tax_price, subtotal.tax_and_service, subtotal.etc, total, total.total_price, total.total_etc, total.cashprice, total.changeprice, total.creditcardprice, total.emoneyprice, total.menutype_cnt, total.menuqty_cnt
                ]

            Instructions:
            - Do not include any explanation.
            - Only respond with a JSON file. 

            Invoice content:
            \"\"\"{row['image_text']}\"\"\""""
            model_output = model.generate_response(prompt)
            print(f"Model output: {model_output}")
            print(f"Concepts: {row['ground_truth']}")

            # Evaluate the model output
            evaluation = evaluate_model_output(
                model_output, json.dumps(row["ground_truth"])
            )
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

        print(
            f"Model: {model_name}, Average BLEU: {avg_bleu}, Average ROUGE: {avg_rouge}, Semantic Accuracy: {semantic_accuracy}"
        )


if __name__ == "__main__":
    main()
