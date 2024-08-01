from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np

def generate_answer_a(question, hf_token):
    tokenizer = AutoTokenizer.from_pretrained('elyza/Llama-3-ELYZA-JP-8B', token=hf_token, legacy=False)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        'elyza/Llama-3-ELYZA-JP-8B',
        torch_dtype="auto",
        device_map="auto",
        token=hf_token
    ).eval()
    
    DEFAULT_SYSTEM_PROMPT = "あなたは誠実で優秀な日本人のアシスタントです。特に指示が無い場合は、常に日本語で回答してください。"
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    token_ids = tokenizer.encode(
        prompt, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=512
    )
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
    answer = tokenizer.decode(
        output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True
    )
    del model
    del tokenizer
    torch.cuda.empty_cache()
    return answer

