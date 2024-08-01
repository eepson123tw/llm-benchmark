import numpy as np
from embedding_utils import calculate_similarity
from model_a import generate_answer_a
from model_b import generate_answer_b
from model_c import generate_answer_c
# from model_d import generate_answer_d

# testing questions and standard answers
questions = ["日本の首都はどこですか？", "富士山の高さは？"]
standard_answers = ["東京です。", "3776メートルです。"]

Model_A = 'elyza/Llama-3-ELYZA-JP-8B'
Model_B = 'rinna/japanese-gpt2-medium'
Model_C = 'line-corporation/japanese-large-lm-3.6b'

# generate answers and calculate similarity scores
similarity_scores = {Model_A: [], Model_B: [], Model_C: []}
generated_answers = {Model_A: [], Model_B: [], Model_C: []}

hf_token = "your_huggingface_api_token"

for question, reference in zip(questions, standard_answers):
    answer_a = generate_answer_a(question, hf_token)
    similarity_scores[Model_A].append(calculate_similarity(answer_a, reference))
    generated_answers[Model_A].append(answer_a)

    answer_b = generate_answer_b(question, hf_token)
    similarity_scores[Model_B].append(calculate_similarity(answer_b, reference))
    generated_answers[Model_B].append(answer_b)

    answer_c = generate_answer_c(question)
    similarity_scores[Model_C].append(calculate_similarity(answer_c, reference))
    generated_answers[Model_C].append(answer_c)

    # answer_d = generate_answer_d(question)
    # similarity_scores['Model D'].append(calculate_similarity(answer_d, reference))
    # generated_answers['Model D'].append(answer_d)

# calculate average scores
average_scores = {model_name: np.mean(scores) for model_name, scores in similarity_scores.items()}
print("Average Scores:", average_scores)

# gerenate markdown table
markdown_content = "# 模型回答及相似度评分\n\n"
for model_name, answers in generated_answers.items():
    markdown_content += f"## Model: {model_name}\n"
    markdown_content += "| Question | Answer | Similarity |\n"
    markdown_content += "| --- | --- | --- |\n"
    for question, answer, similarity in zip(questions, answers, similarity_scores[model_name]):
        # replace newlines and pipes in answers
        answer = answer.replace('\n', ' ').replace('|', ' ')
        markdown_content += f"| {question} | {answer} | {similarity:.4f} |\n"
    markdown_content += "\n"

# save markdown table to file
with open("model_responses.md", "w", encoding="utf-8") as f:
    f.write(markdown_content)

print("Markdown table saved to model_responses.md")
