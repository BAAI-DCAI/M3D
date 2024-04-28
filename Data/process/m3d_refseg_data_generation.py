import os
from tqdm import tqdm
from Qwen.examples.vllm_wrapper import vLLMWrapper
import re
import csv
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

model = vLLMWrapper('PATH/Qwen/Qwen-72B-Chat', tensor_parallel_size=8)
root_path = './Data/data/RefSegData/'

path = "./Data/data/RefSegData/RefSeg_data.csv"
with open(path, "a", newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(
        ["Image", "Mask", "Mask_ID", "Question_Type", "Question", "Answer"])

system = """
You are a medical AI visual assistant that can analyze a single CT image. Unfortunately you can't see the image but you can receive a diagnostic report of a local area in the CT image. The report describes the abnormal lesion in the image.
The task is to use the provided report information to create plausible 6 questions and answers about the image for reasoning segmentation tasks
"""

for root, dirs, files in tqdm(os.walk(root_path)):
    if "text.txt" in files:

        img_path = os.path.join(root, "ct.nii.gz")
        mask_path = os.path.join(root, "mask.nii.gz")
        txt_path = os.path.join(root, "text.json")

        with open(txt_path, "r", encoding="utf-8") as json_file:
            text_data = json.load(json_file)

        for id in text_data:
            text = text_data[id]

            user = f"""
                    Report: {text}
                                        
                    Questions and answers need to be structured from the report. But don’t mention the report in Q&A. The question needs to be about a specific lesion area and requires segmentation of this area. The answer needs to use only one <SEG> symbol to refer to the segmentation area and provide a text explanation. 
                    There are two types of questions: one type of question is answered and segmented based on description information, and the other type of question requires reasoning based on general and medical knowledge to obtain answers and segmentation.
                    
                    Example:
                    1). Description-based
                    Question-1: Please segment where the liver cyst appears in the image. Answer: Sure, it is [SEG] on the upper right side of the liver.
                    2). Reasoning-based
                    Question-1: Can you segment the unusual part in this image and explain why? Answer: Sure, it is [SEG]. In the image, the unusual part is ...
                    Question-2: What can make the woman stand higher? Please output segmentation mask and explain why. Answer: Sure, [SEG]. The woman is standing higher by using ...
                    Question-3: If there are any lesions in the largest human body organ in the image, please segment them. Answer: The largest organ is the liver, where liver tumors are present, and the region is the <SEG>.
                    
                    Desired output format:
                    1). Description-based
                    Question-1: ...? Answer: ...\n
                    Question-2: ...? Answer: ...\n
                    Question-3: ...? Answer: ...\n
                    2). Reasoning-based
                    Question-4: ...? Answer: ...\n
                    Question-5: ...? Answer: ...\n
                    Question-6: ...? Answer: ...\n
                    
                    Please construct a total of 6 sets of question and answer pairs according to the desired format, 3 sets of each type.
                    Using specific descriptions in your questions would ensure others can get an accurate answer.
                    Always ask questions and answer as if directly looking at the image.
                    """

            response, _ = model.chat(query=user, history=None,
                                     system=system)

            pattern = r'Question-(\d+): (.*?) Answer: (.*?)\n'
            matches = re.findall(pattern, response)

            with open(path, "a", newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)

                for match in matches:
                    question_id = match[0]
                    question = match[1].strip()
                    answer = match[2].strip()

                    # print("Question:", question)
                    # print("Answer:", answer)
                    # print("---")

                    if int(question_id) < 4:
                        question_type = str(0)
                    else:
                        question_type = str(1)

                    csvwriter.writerow([img_path, mask_path, id, question_type, question, answer])

