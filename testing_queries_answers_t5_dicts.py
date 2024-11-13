import os
import re
import torch
from transformers import (
                          T5Tokenizer,
                          T5ForConditionalGeneration)
import pandas as pd

device='cuda'
hf_model = 'ai-forever/ruT5-large'

queries_answers_df = pd.read_csv(os.path.join("data", "validate_QueriesAnswers.csv"), sep="\t")
queries_answers_dicts = queries_answers_df.to_dict(orient="records")
# texts = ["Query: " + d["Query"] + " Document: " + d["FastAnswerText"] + " Relevant: " for d in queries_answers_dicts]

tokenizer = T5Tokenizer.from_pretrained(hf_model)
# model = T5ForConditionalGeneration.from_pretrained(os.getcwd(), 't5_val_model')
model = T5ForConditionalGeneration.from_pretrained(os.path.join(os.getcwd(), "models", "checkpoint-5000"))
model = model.to(device)

test_results = []
for num, d in enumerate(queries_answers_dicts):
    text = d["Query"] + " Document: " + d["answer"] + " Relevant: "
    input_ids = tokenizer.encode(text,  return_tensors="pt").to(device)
    outputs=model.generate(input_ids, eos_token_id=tokenizer.eos_token_id, max_length=64, early_stopping=True).to(device)
    outputs_decode = tokenizer.decode(outputs[0][1:])

    outputs_logits=model.generate(input_ids, output_scores=True, return_dict_in_generate=True, 
                                  eos_token_id=tokenizer.eos_token_id, max_length=64, early_stopping=True)
    sigmoid_0 = torch.sigmoid(outputs_logits.scores[0][0])
    print(num, "/", len(queries_answers_dicts), "outputs:", outputs, "outputs decode:", outputs_decode, "sigmoid:", sigmoid_0[2])
    d["MouseRelevant"] = re.sub("</s>", "", outputs_decode)
    d["MouseScore"] = sigmoid_0[2].item()
    test_results.append(d)
    # test_results.append({"QueryAnswer": text, "Relevant": re.sub("</s>", "", outputs_decode), "Sigmoid": sigmoid_0[2].item()})

test_results_df = pd.DataFrame(test_results)
print(test_results_df)
test_results_df.to_csv(os.path.join("results", "validate_QueriesAnswers241113.csv"), sep="\t", index=False)