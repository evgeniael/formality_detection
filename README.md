# Evaluating formality detection models

 This repository contains code for evaluating two different models on the task of classifying a sentence as formal or informal. A report on the methodology, results and findings can be found here ([Formality_Detection_Report.pdf](https://github.com/user-attachments/files/19548601/Formality_Detection_Report.pdf)
).

 The `data` folder contains the original dataset used to create the test set employed in this study.
 The `eval_formal` folder contains all scripts used for: 
 
 (1) preprocessing the raw data (preprocessing scripts can be found in the `datasets` folder), 
 
 (2) running the model predictions, and,

 (3) evaluating the quality of these predictions.

 All outputs of the evaluation pipeline can be found in the `outputs` folder. In order to reproduce the experiments, you will need to first run the `predictions.py` script and then the `evaluate.py` script, both found in the `eval_formal` folder, preferably using `Python 3.11` (which was used during the original experiments). 

 **Remarks:**
 
 (1) To replicate the virtual environment used to run these experiments, you can use the `requirements.txt` file provided. You can create a virtual environment similar to what I used, by navigating to the folder you want to create it in and running the following commands in your terminal (replacing PATH_TO with your local path in which `requirements.txt` is saved at):
 
```
python3.11 -m venv "fdenv"
source fdenv/bin/activate
python3.11 -m pip install -r PATH_TO/requirements.txt
```

 (2) When running the scripts, you will need to pass relevant arguments relevant to the results you want to obtain. To **generate the predictions**, you need to choose the `dataset` you want to investigate (either `answers` or `email` (default)), the `model` you want to use for prediction (for now, the code supports 2 models, `mistral-12b` (which relates to [this](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) model) and `m-deberta`(default; relates to [this](https://huggingface.co/s-nlp/deberta-large-formality-ranker) model), and the type of `prompt` needed for the model (`normal_prompt` (default) for a fine-tuned model need to use for `m-deberta`) or `instruction_prompt` for an instruction-tuned model (use for `mistral-12b`)).

An example, where you need to replace `PATH_TO` with the path to the `formality_detection` folder on your local device:

```
python3.11 PATH_TO/formality_detection/eval_formal/predictions.py --dataset answers --prompt normal_prompt --model m-deberta
```

After running this script, to **obtain the evaluation performance metrics** for the models, you need to pass the previous arguments and an argument that decides how the gold label from the raw datasets are binarised (either `avg_sign`, `maj_sign` or `sample_sign`(default); for details on each, refer to the report). 

Continuing the above example:

```
python3.11 PATH_TO/formality_detection/eval_formal/evaluate.py --dataset answers --prompt normal_prompt --model m-deberta --gold_label_transform avg_sign
```

 (3) If the resulting file generated from running those 2 scripts already exists in the `outputs/predictions` or `outputs/evaluation` folders, the script will not proceed to generate those again (it performs a check whether they already exist). Hence, if you'd like to re-generate those, make sure to delete the existing resulted files in those two folders when you run the scrips on your local device.

