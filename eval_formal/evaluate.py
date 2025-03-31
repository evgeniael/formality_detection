from eval_utils import load_binary_gold_label, compute_metrics

from pathlib import Path
from tqdm import tqdm

import argparse
import json
import pandas as pd
import random
import os
import numpy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="normal_prompt") # normal_prompt instruction_prompt
    parser.add_argument("--model", type=str, default="m-deberta") # m-deberta mistral-12b
    parser.add_argument("--dataset", type=str, default="email") # answers email
    parser.add_argument("--num_datapoints", type=int, default=500)
    parser.add_argument("--gold_label_transform", type=str, default="sample_sign") # avg_sign maj_sign #sample_sign
    args = parser.parse_args()

    random.seed(10)
    # random.seed(20)
    # random.seed(30)

    project_root = Path(__file__).resolve().parent / ".."

    predictions_dir = project_root / "outputs" / "predictions"
    predictions_file = predictions_dir / f"{args.dataset}-{args.num_datapoints}-{args.model}-predictions.parquet"
    
    output_dir = project_root / "outputs" / "evaluation"
    os.makedirs(output_dir, exist_ok=True)

    evaluation_file = output_dir / f"{args.dataset}-{args.num_datapoints}-{args.model}-{args.gold_label_transform}-evaluation.json"

    ### Get model predictions for evaluation dataset
    if not evaluation_file.exists():
        predictions = pd.read_parquet(predictions_file, engine="pyarrow")
        
        preds = []
        gold_labels = []

        evaluation_data = []
        for x, row in tqdm(predictions.iterrows(), total=len(predictions)):
            id = row["id"]
            text = row["text"]
            avg_formal = row["avg_formal"]
            votes_formal = row["votes_formal"].tolist()
            prediction = row["prediction"] 

            row_data = {"id":id, "text":text, "avg_formal":avg_formal, "votes_formal":votes_formal, "prediction":prediction}

            binary_label = load_binary_gold_label(args.gold_label_transform, avg_formal, votes_formal)
            row_data["binary_label"] = binary_label

            if not numpy.isnan(prediction):
                preds.append(prediction)
                gold_labels.append(binary_label)

            evaluation_data.append(row_data)
        
        performance_metrics = compute_metrics(preds, gold_labels)
        print(performance_metrics)

        data = {}
        data["metadata"] = {
            "dataset": args.dataset,
            "model": args.model,
            "prompt": args.prompt,
            "num_datapoints": args.num_datapoints,
            "gold_label_transform": args.gold_label_transform,
            "classification_performance_metrics": performance_metrics
        }

        data["data"] = evaluation_data
        with open(evaluation_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Evaluation data saved to {evaluation_file}.")

if __name__ == "__main__":
    main()

