import base64
import requests
import os
import csv
import pandas as pd
import tqdm

# OpenAI API Key
api_key = ""

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_folders = r"D:\Dataset\test_data\test_data\ct_quizze"
image_folder_list = os.listdir(image_folders)

data_csv_path = r"C:\Users\baifa\Desktop\3DPT\VQA-data\check\VQA_data_mc_test_bench_revised.csv"
data = pd.read_csv(data_csv_path, dtype='str')

close_ended = True

test_res_path = r"D:\Dataset\test_data\test_gpt4v_close_vqa"+str(id)+".csv"
with open(test_res_path, mode='w',newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["Image Path", "Text", "Question Type", "Question", "Choice A", "Choice B", "Choice C", "Choice D", "Answer", "Answer Choice", "Predtion"])
    for index, paires in tqdm.tqdm(data.iterrows()):
        question_type = paires["Question Type"]

        img_path = os.path.join(image_folders, paires["Image Path"])

        if close_ended:
            question = paires["Question"]
            if pd.isna(paires["Choice C"]):
                choice = "Choices: A. {} B. {}".format(paires["Choice A"], paires["Choice B"])
            else:
                choice = "Choices: A. {} B. {} C. {} D. {}".format(paires["Choice A"], paires["Choice B"],
                                                                   paires["Choice C"],
                                                                   paires["Choice D"])
            question = question + ' ' + choice
            answer = "{}. {}".format(paires["Answer Choice"], paires["Answer"])
        else:
            question = paires["Question"]
            answer = str(paires["Answer"])

        instruction = "Please give the answer about the question directly without any other explanation. Do not say sorry!"

        content_list = []
        content_list.append({
            "type": "text",
            "text": instruction + '\n' + question
        })
        num = 5
        files = os.listdir(img_path)
        sorted_files = sorted(files)
        if len(files) < num + 1:
            for file_name in sorted_files:
                file_path = os.path.join(img_path, file_name)
                base64_image = encode_image(file_path)
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
        else:
            step = len(sorted_files) // num
            sampled_files = sorted_files[::step]
            for file_name in sampled_files:
                file_path = os.path.join(img_path, file_name)
                base64_image = encode_image(file_path)
                content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                })

            # Getting the base64 string
            # base64_image = encode_image(image_path)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": content_list
                }
            ],
            "max_tokens": 300
        }

        attemp = 3
        for i in range(attemp):
            try:
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                output = response.json()['choices'][0]['message']['content']
                # print(output)
                writer.writerow(
                    [paires["Image Path"], paires["Text"], paires["Question Type"], paires["Question"], paires["Choice A"], paires["Choice B"], paires["Choice C"], paires["Choice D"], paires["Answer"], paires["Answer Choice"], output])
                break
            except Exception as e:
                print("error attemp.", e)
