import os
from tqdm import tqdm
from examples.vllm_wrapper import vLLMWrapper
import re
import csv
import json

# multiple groups in parallel improves speed
group_id = 0
group_num = 4

model = vLLMWrapper('PATH/Qwen/Qwen-72B-Chat', tensor_parallel_size=2)

root_path = 'PATH/data/M3D_Cap_npy/'
file_path = "PATH/data/3DCTTEXT_npy/M3D_Cap.json"
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)


train_data = data.get('train', [])
total_items = len(train_data)
chunk_size = total_items // group_num

split_train_data = [train_data[i:i + chunk_size] for i in range(0, total_items, chunk_size)]

data_list = split_train_data[group_id]
data_len = len(data_list)
print("data_len: ",data_len)

vqa_data_name = "M3D_VQA_" + str(group_id) + ".csv"
path = "PATH/Qwen/VQA-data/" + vqa_data_name
with open(path, "a", newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(
        ["Image Path", "Text", "Question Type", "Question", "Choice A", "Choice B", "Choice C", "Choice D", "Answer", "Answer Choice"])

system = """
You are a medical AI visual assistant that can analyze a single CT image. You receive the file name of the CT image and the medical diagnosis report. The report describes multiple abnormal lesions in the image.
The task is to use the provided CT image and report information to create plausible 9 questions about the image.
Each question corresponds to four options, and these questions come from the following 5 aspects:
1). Planes (axial, sagittal, coronal);
2). CT phase (non-contrast, contrast, arterial phase, portal venous phase, venous phase, delayed phase, parenchymal phase, renal cortical phase, dual phase, renal excretory phase, mixed arteriovenous, myelography, etc.) or window ( bone, lung, window, etc.);
3). Organ;
4). Abnormality type or description;
5). Abnormality position;
"""

for i, data in tqdm(enumerate(data_list)):
    try:
        image_file = os.path.basename(data["image"])

        with open(data["text"], "r") as f:
            text = f.read()

        user = f"""
                Image: {image_file}
                Report: {text}
    
                Desired format:
                1). Planes
                Question-1: ...? Choice: A. ... B. ... C. ... D. ... Answer: A. ...\n
    
                2). CT phase
                Question-2: ...? Choice: A. ... B. ... C. ... D. ... Answer: A. ...\n
    
                3). Organ
                Question-3: ...? Choice: A. ... B. ... C. ... D. ... Answer: A. ...\n
    
                4). Abnormality type or description
                Question-4: ...? Choice: A. ... B. ... C. ... D. ... Answer: A. ...\n
    
                Question-5: ...? Choice: A. ... B. ... C. ... D. ... Answer: A. ...\n
    
                Question-6: ...? Choice: A. ... B. ... C. ... D. ... Answer: A. ...\n
    
                5). Abnormality position
                Question-7: ...? Choice: A. ... B. ... C. ... D. ... Answer: A. ...\n
    
                Question-8: ...? Choice: A. ... B. ... C. ... D. ... Answer: A. ...\n
    
                Question-9: ...? Choice: A. ... B. ... C. ... D. ... Answer: A. ...\n
    
                Make the correct answers randomly distributed among the four choices.
                If there is a true or false question, please ensure that the proportion of yes and no is equivalent. For example, Is ... ? Are ... ?, Do ... ?, Does ... ?, Did ... ?, Can ... ?.
                Please do NOT ask directly what organs or abnormalities are visible in the image, as the answers are not unique. It would be best to use specific descriptions in your questions to ensure that other people can get an accurate answer even without providing choices.
    
                Please be careful not to mention the file name and report. Always ask questions and answer as if directly looking at the image.
                """

        response, _ = model.chat(query=user, history=None,
                                 system=system)

        questions = re.findall(r'Question-(\d+): (.*?)(?: Choice: (.*?))? Answer: ([A-D])\. (.*?)\n', response)

        with open(path, "a", newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)

            for q in questions:
                question_num, question, choices, an_choice, answer = q
                choices = re.findall(r"([A-D])\. (.+?)(?=(?: [A-D]\.|$))", choices)
                choices_dict = {choice[0]: choice[1] for choice in choices}

                for option in ['A', 'B', 'C', 'D']:
                    if option not in choices_dict:
                        choices_dict[option] = 'NA'

                if int(question_num) < 4:
                    question_type = question_num
                elif int(question_num) < 7:
                    question_type = str(4)
                else:
                    question_type = str(5)

                csvwriter.writerow(
                    [data["image"], text, question_type, question, choices_dict['A'], choices_dict['B'], choices_dict['C'], choices_dict['D'],
                     answer, an_choice])
    except:
        print("Error in " + "id:" + str(i) + " " + data["image"])
