from predict_utils import load_dataset, load_model, load_tokenizer, load_prompt, get_prediction

from pathlib import Path
from tqdm import tqdm

import argparse
import torch
import pandas as pd
import random
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="normal_prompt") # normal_prompt instruction_prompt
    parser.add_argument("--model", type=str, default="m-deberta") # m-deberta mistral-12b
    parser.add_argument("--dataset", type=str, default="email") # answers email
    parser.add_argument("--num_datapoints", type=int, default=500)
    args = parser.parse_args()

    random.seed(10)

    dataset = load_dataset(args.dataset, args.num_datapoints)

    project_root = Path(__file__).resolve().parent / ".."
    output_dir = project_root / "outputs" / "predictions"
    os.makedirs(output_dir, exist_ok=True)
    predictions_file = output_dir / f"{args.dataset}-{args.num_datapoints}-{args.model}-predictions.parquet"
    
    ### Get model predictions for evaluation dataset
    if not predictions_file.exists():

        model = load_model(args.model)
        tokenizer = load_tokenizer(args.model)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = model.to(device)
        
        prediction_data = []
        print(f"Generating on the {args.dataset} dataset...")
        for idx, row in tqdm(dataset.data.iterrows(), total=len(dataset.data)):
            text = row["text"]
            id = row["id"]
            avg_formal = row["avg_formal"]
            votes_formal = row["votes_formal"]

            model_inputs = load_prompt(args.prompt, text, tokenizer)
            model_inputs = model_inputs.to(device)

            prediction = get_prediction(args.prompt, model_inputs, model, tokenizer)

            prediction_data.append({
                "id": id,
                "text": text,
                "avg_formal": avg_formal, 
                "votes_formal": votes_formal,
                "prediction": prediction,
            })

        # Convert the data into a dataframe 
        dataframe = pd.DataFrame(prediction_data)
        dataframe.attrs["dataset"] = args.dataset
        dataframe.attrs["model"] = args.model
        dataframe.attrs["prompt"] = args.prompt
        dataframe.attrs["num_datapoints"] = args.num_datapoints

        # Store the data
        dataframe.to_parquet(predictions_file, engine="pyarrow", index=False)
        print(f"Data saved to {predictions_file}.")

if __name__ == "__main__":
    main()
