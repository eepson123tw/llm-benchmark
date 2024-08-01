# AI Language Model Comparison Project

This project aims to compare different AI language models for answering questions in Japanese. The comparison is based on the similarity of the model-generated answers to the reference answers using embedding-based similarity metrics.

## Project Overview

The main goal of this project is to evaluate the performance of various AI language models in providing accurate answers to specific questions. The models being compared are:

- `elyza/Llama-3-ELYZA-JP-8B`
- `rinna/japanese-gpt2-medium`
- `line-corporation/japanese-large-lm-3.6b`

The project involves:

1. Generating answers using different AI language models.
2. Calculating the similarity between the generated answers and the reference answers.
3. Comparing the performance of the models based on the similarity scores.

## Dependencies

The project requires the following Python packages:

- `torch`
- `transformers`
- `numpy`

You can install these dependencies using the `requirements.txt` file.

```sh
pip install -r requirements.txt
```

````

## Usage

### Generating `requirements.txt`

To generate the `requirements.txt` file from your current environment, use:

```sh
pip freeze > requirements.txt
```

### Running the Project

1. **Set up your environment**:

   - Ensure you have Python installed.
   - Create and activate a virtual environment (optional but recommended).

   ```sh
   python -m venv venv
   source venv/bin/activate  # For Unix or MacOS
   # or
   .\venv\Scripts\activate  # For Windows
   ```

2. **Install dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

3. **Run the main script**:

   - Make sure you have your Hugging Face API token.
   - Replace `your_huggingface_api_token` with your actual token in the script.

   ```sh
   python main.py
   ```

### Script Structure

- `main.py`: Main script to run the model comparison.
- `embedding_utils.py`: Utility functions to calculate embedding-based similarity.
- `model_a.py`, `model_b.py`, `model_c.py`, `model_d.py`: Scripts to generate answers from different models.
- `requirements.txt`: List of dependencies.

### Generating Answers and Calculating Similarity

The `main.py` script performs the following steps:

1. Loads each model and tokenizer.
2. Generates answers for a set of predefined questions.
3. Calculates the similarity of the generated answers to the reference answers.
4. Outputs the results in a markdown table format.

## Example Output

The example output includes a markdown table with questions, model-generated answers, and their similarity scores.

```markdown
# Model Answers and Similarity Scores

## Model: elyza/Llama-3-ELYZA-JP-8B

| Question                 | Answer                                        | Similarity |
| ------------------------ | --------------------------------------------- | ---------- |
| 日本の首都はどこですか？ | という質問に「東京」と答えるのと同じです。... | 0.7837     |
| 富士山の高さは？         | 富士山の高さは、3,776 メートルです。...       | 0.8031     |

## Model: rinna/japanese-gpt2-medium

| Question                 | Answer                                                                                                                                                       | Similarity |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| 日本の首都はどこですか？ | _ q&a ページ _ q&a _ サポート・お問い合わせ _ ソニー \_ ...                                                                                                  | 0.5881     |
| 富士山の高さは？         | 富士山は、日本の国土のほぼ中央に位置し、日本百名山の一つに数えられる山です。 富士山は、日本の国土のほぼ中央に位置し、日本百名山の一つに数えられる山です。... | 0.6866     |

## Model: line-corporation/japanese-large-lm-3.6b

| Question                 | Answer                                                                    | Similarity |
| ------------------------ | ------------------------------------------------------------------------- | ---------- |
| 日本の首都はどこですか？ | 日本の首都はどこですか？ 商品 本文: K18YG イエローゴールド パール 真珠... | 0.6993     |
| 富士山の高さは？         | 富士山の高さは？ 商品 本文: K18YG イエローゴールド パール 真珠...         | 0.7422     |
```

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.


````
