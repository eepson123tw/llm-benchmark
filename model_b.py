from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

def generate_answer_b(question, hf_token):
    tokenizer = AutoTokenizer.from_pretrained('rinna/japanese-gpt2-medium', token=hf_token, legacy=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        'rinna/japanese-gpt2-medium',
        torch_dtype="auto",
        device_map="auto",
        token=hf_token
    ).eval()

    token_ids = tokenizer.encode(question, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=512)
    attention_mask = token_ids.ne(tokenizer.pad_token_id).long()

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_new_tokens=200,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    answer = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True)
    del model
    torch.cuda.empty_cache()
    return answer
