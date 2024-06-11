import os
from tqdm import tqdm
from Qwen.examples.vllm_wrapper import vLLMWrapper

import re
import csv
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

id = 0
world_size = 1

model = vLLMWrapper('Qwen/Qwen-72B-Chat', tensor_parallel_size=2)

in_eval_path = "./LaMed/output/test_caption_linear.csv"
out_eval_path = "./LaMed/output/test_caption_linear_eval.csv"

system = """
You are an AI assistant, please evaluate based on the following.
"""

with open(in_eval_path, "r", newline='', encoding='latin') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    data = list(csvreader)

    total_items = len(data)
    chunk_size = total_items // world_size
    split_train_data = [data[i:i + chunk_size] for i in range(0, total_items, chunk_size)]

    data_list = split_train_data[id]
    data_len = len(data_list)
    print("data_len: ", data_len)

    for row in tqdm(data_list):
        answer = row[1]
        pred = row[2]

        user = f"""
                Please refer to the ground truth and prediction based on the following two paragraphs, identify the aspects mentioned in the ground truth, and calculate the percentage of these aspects that are either correctly mentioned or partially matched in the prediction, scoring from 0 to 100.
                ground truth: {answer}
                prediction: {pred}
                
                The output format is:
                Score: xx.
                """

        response, _ = model.chat(query=user, history=None,
                                 system=system)

        with open(out_eval_path, "a", newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)

            pattern = r"Score: (\d+\.\d+)"

            match = re.search(pattern, response)

            if match:
                score = match.group(1)
            else:
                score = 'NA'

            csvwriter.writerow(
                [score])
