import os
import sys
from tqdm import tqdm
from Qwen.examples.vllm_wrapper import vLLMWrapper
import csv
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

group_id = 0
group_num = 4

model = vLLMWrapper('PATH/Qwen/Qwen-72B-Chat', tensor_parallel_size=2)

path1 = "./Data/data/M3D-VQA/M3D_VQA_test.csv"
path2 = "./Data/data/M3D-VQA/M3D_VQA_test_checked.csv"

system = """
You are a medical AI assistant. Please provide answers and help based on the following questions.
"""

with open(path1, "r", newline='', encoding='utf-8') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    data = list(csvreader)

    total_items = len(data)
    chunk_size = total_items // group_num
    split_train_data = [data[i:i + chunk_size] for i in range(0, total_items, chunk_size)]

    data_list = split_train_data[group_id]
    data_len = len(data_list)
    print("data_len: ", data_len)

    for row in tqdm(data_list):
        img_path = row[0]
        qustion_type = row[2]
        report = row[1]
        question = row[3]
        choiceA = row[4]
        choiceB = row[5]
        choiceC = row[6]
        choiceD = row[7]
        answer = row[8]
        answer_choice = row[9]

        user = f"""
                This is a question from the visual question-answering dataset. Questions are generated based on information from image and report. The generated data inevitably contains certain errors. 
                Please use the following information to determine whether the content described in the question is consistent with the text report and whether the answer is correct.
                Image Path: {img_path}
                Report: {report}
                Question: {question}
                Choices: A.{choiceA} B.{choiceB} C.{choiceC} D.{choiceD}
                Answer Choice: {answer_choice}. {answer}
                
                If there is an error, please answer ’NO‘ first and give a more reasonable question and answer. If it is basically correct, answer 'Yes' directly. Do not give redundant answers.
                """

        response, _ = model.chat(query=user, history=None,
                                 system=system)

        with open(path2, "a", newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)

            if "Yes" in response:
                crt = 1
            elif "No" in response:
                crt = 0
            else:
                crt = -1

            csvwriter.writerow(
                [img_path, report, qustion_type, question, choiceA, choiceB,choiceC,choiceD,answer,answer_choice,crt])
