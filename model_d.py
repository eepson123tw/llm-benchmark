from transformers import LlamaTokenizer, AutoModelForCausalLM
import torch

def generate_answer_d(question):
    tokenizer = LlamaTokenizer.from_pretrained("novelai/nerdstash-tokenizer-v1", additional_special_tokens=['▁▁'], legacy=False)
    model = AutoModelForCausalLM.from_pretrained(
        "stabilityai/japanese-stablelm-base-alpha-7b",
        trust_remote_code=True,
    )
    model.half()
    model.eval()

    # use GPU if available
    device = "cpu"

    input_ids = tokenizer.encode(question, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=512)
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

    seed = 23
    torch.manual_seed(seed)

    tokens = model.generate(
        input_ids.to(device=device),
        attention_mask=attention_mask.to(device),
        max_new_tokens=128,
        temperature=1,
        top_p=0.95,
        do_sample=True,
    )

    answer = tokenizer.decode(tokens[0], skip_special_tokens=True)
    del model
    torch.cuda.empty_cache()
    return answer
