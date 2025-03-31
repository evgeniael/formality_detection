from datasets import Answers, Email
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
import torch

def load_dataset(name, num_datapoints):
    name = name.lower()
    if name == "answers":
        return Answers(num_datapoints)
    elif name == "email":
        return Email(num_datapoints)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def load_tokenizer(name):
    name = name.lower()
    if name == "m-deberta":
        return AutoTokenizer.from_pretrained('s-nlp/deberta-large-formality-ranker')
    elif name == "mistral-12b":
        return AutoTokenizer.from_pretrained('mistralai/Mistral-Nemo-Instruct-2407')
    else: 
        raise ValueError(f"Unknown model: {name}")

def load_model(name):
    name = name.lower()
    if name == "m-deberta":
        return AutoModelForSequenceClassification.from_pretrained('s-nlp/deberta-large-formality-ranker')
    elif name == "mistral-12b":
        return AutoModelForCausalLM.from_pretrained('mistralai/Mistral-Nemo-Instruct-2407', torch_dtype=torch.bfloat16)
    else: 
        raise ValueError(f"Unknown model: {name}")

def load_prompt(name, string, tokenizer):
    name = name.lower()
    if name == "instruction_prompt":
        p = f"You are provided with a string. Predict whether the string is expressed in formal or informal language. Just generate formal or informal. \nString: '{string}' \nPrediction:"
        # print(p)
        prompt = [{"role": "user", "content": p}]
        inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True)
        return inputs
    elif name == "normal_prompt":
        inputs = tokenizer(string, return_tensors="pt")
        return inputs
    else:
        raise ValueError(f"Unknown prompting method: {name}")

def get_prediction(name, inputs, model, tokenizer):
    name = name.lower()
    if name == "instruction_prompt":
        generation_ids = model.generate(inputs, num_beams=1, do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
        
        prompt_length = inputs.shape[1]
        generation = tokenizer.decode(generation_ids[0][prompt_length:], skip_special_tokens=True)
        if 'informal' in generation.lower():
            return 0
        elif 'formal' in generation.lower():
            return 1
        else:
            return None
    elif name == "normal_prompt":
        with torch.no_grad():      
            outputs = model(inputs['input_ids'], inputs['attention_mask'])
        logits = outputs.logits
        _, prediction = torch.max(logits, dim=1)
        return 1 - prediction.item()
    else:
        raise ValueError(f"Unknown prompting method: {name}")
