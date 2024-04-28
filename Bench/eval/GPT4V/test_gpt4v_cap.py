import base64
import requests
import os
import csv

# OpenAI API Key
api_key = ""

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_folders = r"D:\Dataset\test_data\test_data\ct_quizze"
image_folder_list = os.listdir(image_folders)

test_res_path = r"D:\Dataset\test_data\test_gpt4v_caption.csv"
with open(test_res_path, mode='w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["Question", "Ground Truth", "pred"])

    for image_folder in image_folder_list:
        content_list = []
        content_list.append({
            "type": "text",
            "text":  "What are the findings and abnormality of these images?"
        })
        num = 10
        folders = os.listdir(os.path.join(image_folders,image_folder))
        for folder in folders:
            folder_path = os.path.join(image_folders,image_folder,folder)
            files = os.listdir(folder_path)
            sorted_files = sorted(files)
            if len(files) < num+1:
                for file_name in sorted_files:
                    file_path = os.path.join(folder_path, file_name)
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
                    file_path = os.path.join(folder_path, file_name)
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

            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

            response_json = response.json()
            output = response_json['choices'][0]['message']['content']
            print(output)
            writer.writerow(
                ["Please provide a caption consists of findings for this medical image.",
                 " ",
                 output])
