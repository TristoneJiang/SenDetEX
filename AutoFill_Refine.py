import random
import openai
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math
gamma = 0.35
m = {"xsum": 5000, "writing_prompts": 3500}
temperature = 0.7
seed = 42
random.seed(seed)
oracle_model_name = "xxx"
tokenizer = AutoTokenizer.from_pretrained(oracle_model_name)
model = AutoModelForCausalLM.from_pretrained(oracle_model_name, device_map="auto").eval()
def compute_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    return math.exp(loss)
def call_generator(text, model_name):
    prompt = f"Fill in each [MASK] in the following document with a single sentence to ensure overall fluency, coherence, and logic. Original document: {text}. New completed document:"
    if model_name == "gpt-4o":
        openai.api_key = "xxx"
        openai.api_base = "xxx"
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response["choices"][0]["message"]["content"].strip()
    elif model_name == "deepseek":
        openai.api_key = "xxx"
        openai.api_base = "xxx"
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response["choices"][0]["message"]["content"].strip()
    else:
        raise ValueError("Unsupported model name.")
def random_mask_sentences(document, gamma=0.35):
    sentences = document.split('. ')
    num_to_mask = max(1, int(len(sentences) * gamma))
    mask_indices = random.sample(range(len(sentences)), num_to_mask)
    for i in mask_indices:
        sentences[i] = "[MASK]"
    return '. '.join(sentences)
def process_dataset(dataset_name, generator_name, max_samples):
    dataset = load_dataset(dataset_name, split='train')
    selected_data = dataset.shuffle(seed=seed).select(range(max_samples))
    d_ori, d_masked, d_re = [], [], []
    for example in selected_data:
        doc = example['document'] if dataset_name == 'xsum' else example['prompt'] + ' ' + example['story']
        masked_doc = random_mask_sentences(doc, gamma=gamma)
        try:
            generated_doc = call_generator(masked_doc, generator_name)
            ppl_original = compute_perplexity(doc)
            ppl_generated = compute_perplexity(generated_doc)
            if ppl_generated < ppl_original:
                d_ori.append(doc)
                d_masked.append(masked_doc)
                d_re.append(generated_doc)
        except Exception as e:
            print(f"Generation failed: {e}")
    return d_ori, d_masked, d_re
XD = process_dataset('xsum', 'deepseek', m['xsum'])
XG = process_dataset('xsum', 'gpt-4o', m['xsum'])
WD = process_dataset('writing_prompts', 'deepseek', m['writing_prompts'])
WG = process_dataset('writing_prompts', 'gpt-4o', m['writing_prompts'])
def split_dataset(dataset, ratios=(0.6, 0.2, 0.2)):
    total = len(dataset)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    return dataset[:train_end], dataset[train_end:val_end], dataset[val_end:]
splits = {}
for name, data in zip(['XD', 'XG', 'WD', 'WG'], [XD, XG, WD, WG]):
    splits[name] = {
        'train': split_dataset(data[2])[0],
        'val': split_dataset(data[2])[1],
        'test': split_dataset(data[2])[2]
    }