from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
import torch

def generate_answer_c(question):
    tokenizer = AutoTokenizer.from_pretrained('line-corporation/japanese-large-lm-3.6b', use_fast=False)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model = AutoModelForCausalLM.from_pretrained('line-corporation/japanese-large-lm-3.6b', torch_dtype=torch.float16).to('cuda')
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    set_seed(101)
    outputs = generator(
        question,
        max_length=200,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,
    )
    answer = outputs[0]['generated_text']
    del model
    torch.cuda.empty_cache()
    return answer
