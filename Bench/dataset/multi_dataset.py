import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset, ConcatDataset

import json
import pandas as pd

import monai.transforms as mtf
from monai.data import load_decathlon_datalist
from monai.data import set_track_meta

from ..utils import mask2box
from .dataset_info import dataset_info



class ITRDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def truncate_text(self, input_text, max_tokens):
        def count_tokens(text):
            tokens = self.tokenizer.encode(text, add_special_tokens=True)
            return len(tokens)

        if count_tokens(input_text) <= max_tokens:
            return input_text

        sentences = input_text.split('.')

        selected_sentences = []
        current_tokens = 0

        if sentences:
            selected_sentences.append(sentences.pop(0))

        while current_tokens <= max_tokens and sentences:
            random_sentence = random.choice(sentences)
            new_tokens_len = count_tokens(random_sentence)
            if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
                selected_sentences.append(random_sentence)
                current_tokens += new_tokens_len
            else:
                sentences.remove(random_sentence)

        truncated_text = '.'.join(selected_sentences)
        return truncated_text

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)

                image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)

                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, 'r') as text_file:
                    raw_text = text_file.read()
                text = self.truncate_text(raw_text, self.args.max_length)

                text_tensor = self.tokenizer(
                    text, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                ret = {
                    'image': image,
                    'text': text,
                    'input_id': input_id,
                    'attention_mask': attention_mask,
                    'question_type': "Image_text_retrieval",
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)




class CapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]

        self.caption_prompts = [
            "Can you provide a caption consists of findings for this medical image?",
            "Describe the findings of the medical image you see.",
            "Please caption this medical scan with findings.",
            "What is the findings of this image?",
            "Describe this medical scan with findings.",
            "Please write a caption consists of findings for this image.",
            "Can you summarize with findings the images presented?",
            "Please caption this scan with findings.",
            "Please provide a caption consists of findings for this medical image.",
            "Can you provide a summary consists of findings of this radiograph?",
            "What are the findings presented in this medical scan?",
            "Please write a caption consists of findings for this scan.",
            "Can you provide a description consists of findings of this medical scan?",
            "Please caption this medical scan with findings.",
            "Can you provide a caption consists of findings for this medical scan?"
        ]

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif mode == 'test':
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list[idx]
                image_path = data["image"]
                image_abs_path = os.path.join(self.data_root, image_path)

                image = np.load(image_abs_path)  # nomalized 0-1, C,D,H,W
                # image = np.load(img_abs_path)[np.newaxis, ...]  # nomalized
                image = self.transform(image)

                text_path = data["text"]
                text_abs_path = os.path.join(self.data_root, text_path)
                with open(text_abs_path, 'r') as text_file:
                    raw_text = text_file.read()
                answer = raw_text

                prompt_question = random.choice(self.caption_prompts)

                question = self.image_tokens + prompt_question

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[label == self.tokenizer.pad_token_id] = -100
                label[:question_len] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "Caption",
                }
                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class VQADataset(Dataset):
    def __init__(self, args, tokenizer, close_ended=True, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.close_ended = close_ended

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if mode == "train":
            self.data_list = pd.read_csv(args.vqa_data_train_path)
        elif mode == "validation":
            self.data_list = pd.read_csv(args.vqa_data_val_path, nrows=2048)
        elif mode == "test":
            self.data_list = pd.read_csv(args.vqa_data_test_path)
        else:
            print("The mode is not desired ! ")

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlip(prob=0.10, spatial_axis=0),
                mtf.RandFlip(prob=0.10, spatial_axis=1),
                mtf.RandFlip(prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),

                mtf.ToTensor(dtype=torch.float),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif mode == 'test':
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_abs_path = os.path.join(self.args.data_root, data["Image Path"])

                image = np.load(image_abs_path)  # nomalized, 0-1, C,D,H,W
                # image = np.load(img_path)[np.newaxis, ...]  # nomalized

                image = self.transform(image)

                if self.close_ended:
                    question = data["Question"]
                    choices = "Choices: A. {} B. {} C. {} D. {}".format(data["Choice A"], data["Choice B"], data["Choice C"], data["Choice D"])
                    question = question + ' ' + choices
                    answer = "{}. {}".format(data["Answer Choice"], data["Answer"])
                else:
                    question = data["Question"]
                    answer = str(data["Answer"])


                question = self.image_tokens + ' ' + question
                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[label == self.tokenizer.pad_token_id] = -100
                label[:question_len] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'answer_choice': data["Answer Choice"],
                    'question_type': data["Question Type"],
                }

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class PosRECDataset(Dataset):
    def __init__(self, args, tokenizer, tag="0000", description=True, mode='train'):
        self.args = args
        self.tokenizer = tokenizer

        self.tag = tag
        self.mode = mode
        self.description = description

        self.dataset_info = dataset_info

        self.image_tokens = "<im_patch>" * args.proj_out_num
        self.box_tokens = ["<bx_start>", "<bx_end>"]

        root_path = args.seg_data_path
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="training",
            )
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="validation",
            )
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="validation",
            )

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif mode == 'test':
            self.transform = val_transform

        with open(self.args.term_dict_path, 'r') as file:
            self.term_dict = json.load(file)

        self.question_list = [
            "Can you find the {} in this image? Give coordinates.",
            "Can you find {} in this image? Please output the coordinates.",
            "Please bounding the {} by box in this image.",
            "Where is {} in this image? Please respond with a bounding box.",
            "Where is {} in this image? Please output the box.",
            "Can you locate the {} in this image? Please output its coordinates.",
            "Could you mark the {} by bounding box in this image?",
            "Where can I find the {} in this image? Please provide its bounding box.",
            "Identify the indicated {} in this image. Please provide the coordinates of its bounding box.",
        ]

        self.answer_list = [
            "Coordinates are {}.",
            "Sure, {}.",
            "Sure, it is {}.",
            "Sure, the bounding box is {}.",
            "{}.",
            "Here are the coordinates: {}.",
            "Of course, it's located at {}.",
            "The bounding box is given by {}.",
            "The box is {}.",
        ]

        self.description_list = [
            "Description: {} Please answer and find it by box based on the above description.",
            "Definition: {} Please answer and show the bounding box based on the above definition.",
            "Description: {} Can you answer and find it by coordinates based on the above description.",
            "Definition: {} Please output the bounding box and answer based on the above definition.",
            "Description: {} Respond and locate it using a bounding box according to the description.",
            "Definition: {} Please provide an answer and display the bounding box according to the given definition.",
            "Description: {} Can you identify and locate it by coordinates, following the provided description or definition?",
            "Definition: {} Please output the bounding box and provide an answer based on the provided definition.",
            "Based on the description or definition, please respond to {} and indicate its location with a bounding box.",
            "{} Please answer and find it by box based on the above description.",
            "{} Please answer and show the bounding box based on the above definition.",
            "{} Can you answer and find it by coordinates based on the above description.",
            "{} Please output the bounding box and answer based on the above definition.",
            "{} Respond and locate it using a bounding box according to the description.",
            "{} Please provide an answer and display the bounding box according to the given definition.",
            "{} Can you identify and locate it by coordinates, following the provided description or definition?",
            "{} Please output the bounding box and provide an answer based on the provided definition.",
            "Please answer and find it by box based on the description. {}",
            "Please answer and show the bounding box based on the definition. {}",
            "Can you answer and find it by coordinates based on the description. {}",
            "Please output the bounding box and answer based on the definition. {}",
            "Respond and locate it using a bounding box according to the description. {}",
            "Please provide an answer and display the bounding box according to the given definition. {}",
            "Can you identify and locate it by coordinates, following the provided description or definition? {}",
            "Please output the bounding box and provide an answer based on the provided definition. {}",
        ]

        self.answer_cls_list = [
            "The target is {} and the coordinates is {}.",
            "The category is {} and the bounding box is {}.",
            "It is {}, {}.",
            "{}, {}",
            "The target is identified as {} and its coordinates are {}.",
            "The category is {}, the bounding box is provided as {}.",
            "It is characterized by {}, with coordinates {}.",
            "The identified attributes are {}, {}.",
            "Describing it as {}, the corresponding box is {}.",
        ]

        self.answer_no_cls_list = [
            "Sorry, there is no {}",
            "No, we can not see {}",
            "{} is not here",
            "This is {}, but not here.",
            "This is {}, however we can not see it.",
        ]


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_path = data['image']
            seg_path = data['mask']

            image_array = np.load(image_path) #1*32*256*256, normalized
            seg_array = np.load(seg_path)
            cls_id = int(os.path.basename(seg_path).split('_')[1].split('.')[0])

            try:
                item = {
                    'image': image_array,
                    'seg': seg_array,
                }

                it = self.transform(item)

                image = it['image']
                seg = it['seg']  # 1*D*H*W

                cls_list = self.dataset_info[self.tag]
                vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()

                if vld_cls:
                    box = mask2box(seg[0])
                    if not self.description:
                        question_temple = random.choice(self.question_list)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        answer = random.choice(self.answer_list).format(box_text)
                    else:
                        question_temple = random.choice(self.description_list)
                        question = question_temple.format(random.choice(self.term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        answer = random.choice(self.answer_cls_list).format(cls_list[cls_id], box_text)
                else:
                    cls_id = random.randint(0, len(cls_list)-1)
                    if not self.description:
                        question_temple = random.choice(self.question_list)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.answer_no_cls_list).format(cls_list[cls_id])
                    else:
                        question_temple = random.choice(self.description_list)
                        question = question_temple.format(random.choice(self.term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.answer_no_cls_list).format(cls_list[cls_id])

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length",
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[label == self.tokenizer.pad_token_id] = -100
                label[:question_len] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "REC",
                    'tag': self.tag,
                }

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class PosREGDataset(Dataset):
    def __init__(self, args, tokenizer, tag="0000", description=True, mode='train'):
        self.args = args
        self.tokenizer = tokenizer

        self.tag = tag
        self.mode = mode
        self.description = description

        self.dataset_info = dataset_info

        self.image_tokens = "<im_patch>" * args.proj_out_num
        self.box_tokens = ["<bx_start>", "<bx_end>"]

        root_path = args.seg_data_path
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="training",
            )
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="validation",
            )
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="validation",
            )

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif mode == 'test':
            self.transform = val_transform


        with open(args.term_dict_path, 'r') as file:
            self.term_dict = json.load(file)

        self.question_list = [
            "What target is present within the coordinates {} ?",
            "Does the bounding box {} contain any target?",
            "Within the specified region {}, what target is present?",
            "Do you know what it is in the bounding box {}?",
            "What is it in this region {}?",
            "What object is located within the coordinates {}?",
            "Within the specified area {}, what object can be found?",
            "Can you identify the object within the bounding box {}?",
            "What object is present in this region {}?",
        ]
        self.answer_list = [
            "The target is {}.",
            "Sure, the bounding box contains {}.",
            "Sure, it is {}.",
            "Sure, {} is in the bounding box.",
            "{}.",
            "The object is {}.",
            "Of course, it's {}.",
            "Certainly, {} can be found in the bounding box.",
            "Yes, the bounding box includes {}.",
        ]

        self.description_list = [
            "Please describe the target and its function based on the box {} in the image.",
            "Do you know what is it in this bounding box {}? Answer and explain it.",
            "What's the target in the bounding box {}? What function does it have?",
            "What is the area marked with a box {} in the image? Can you explain it?",
            "Could you describe the object and its purpose within the bounding box {} in the image?",
            "Can you identify and describe the object within this bounding box {}? Please explain.",
            "What is the object located in the bounding box {}? Could you explain its function?",
            "Could you describe the area outlined by the box {} in the image? Please explain its significance.",
        ]
        self.answer_cls_list = [
            "Sure, it is {}. {}",
            "The category is {}. {}.",
            "It is {}, {}.",
            "{}, {}",
            "The target is identified as {} and its description is {}.",
            "The category is {}. Description: {}.",
            "It is characterized by {}, {}.",
            "The identified attributes are {}, {}.",
            "Sure, it is {}. Describing it as {}."
        ]
        self.answer_no_cls_list = [
            "Sorry, this is no {}",
            "No, we can not see {}",
            "{} is not here",
            "This is {}, but not here.",
            "This is {}, however we can not see it.",
        ]


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_path = data['image']
            seg_path = data['mask']

            image_array = np.load(image_path) #1*32*256*256, normalized
            seg_array = np.load(seg_path)
            cls_id = int(os.path.basename(seg_path).split('_')[1].split('.')[0])


            try:
                item = {
                    'image': image_array,
                    'seg': seg_array,
                }

                it = self.transform(item)
                image = it['image']
                seg = it['seg']  # 1*D*H*W

                cls_list = self.dataset_info[self.tag]
                vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()

                if vld_cls:
                    box = mask2box(seg[0])
                    if not self.description:
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        question_temple = random.choice(self.question_list)
                        question = question_temple.format(box_text)
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.answer_list).format(cls_list[cls_id])
                    else:
                        box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
                        question_temple = random.choice(self.description_list)
                        question = question_temple.format(box_text)
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.answer_cls_list).format(cls_list[cls_id], random.choice(self.term_dict[cls_list[cls_id]]))
                else:
                    cls_id = random.randint(0, len(cls_list)-1)
                    if not self.description:
                        question_temple = "Can you find the {} in this image? Give coordinates."
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.answer_no_cls_list).format(cls_list[cls_id])
                    else:
                        question_temple = "Description: {} Please answer and find it by box based on the above description."
                        question = question_temple.format(random.choice(self.term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.answer_no_cls_list).format(cls_list[cls_id])

                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length",
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[label == self.tokenizer.pad_token_id] = -100
                label[:question_len] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "REG",
                    'tag': self.tag,
                }

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)



class SegDataset(Dataset):
    def __init__(self, args, tokenizer, tag="0000", description=False, mode='train'):
        self.args = args
        self.tokenizer = tokenizer

        self.tag = tag
        self.description = description
        self.mode = mode
        self.dataset_info = dataset_info

        self.image_tokens = "<im_patch>" * args.proj_out_num

        root_path = args.seg_data_path
        if mode == "train":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="training",
            )
        elif mode == "validation":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="validation",
            )
        elif mode == "test":
            self.data_list = load_decathlon_datalist(
                base_dir=root_path,
                data_list_file_path=os.path.join(root_path, tag, f'{tag}.json'),
                is_segmentation=True,
                data_list_key="validation",
            )

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif mode == 'test':
            self.transform = val_transform

        with open(args.term_dict_path, 'r') as file:
            self.term_dict = json.load(file)

        self.question_list = [
            "Can you segment the {} in this image?",
            "Can you segment {} in this image? Please output the mask.",
            "Please segment the {} in this image.",
            "What is {} in this image? Please respond with segmentation mask.",
            "What is {} in this image? Please output segmentation mask.",
            "Could you provide a segmentation for the {}?",
            "I need the {} segmented from this image.",
            "Segment {} from this image and provide the mask, please.",
            "Please provide a segmentation mask for the {} in this image.",
            "Can you identify and segment the {} in this image?",
        ]

        self.answer_list = [
            "It is [SEG].",
            "Sure, [SEG].",
            "Sure, it is [SEG].",
            "Sure, the segmentation result is [SEG].",
            "[SEG].",
            "The segmentation indicates [SEG].",
            "According to the segmentation, it is [SEG].",
            "The segmentation reveals [SEG].",
            "The segmentation suggests [SEG].",
            "From the segmentation, it appears to be [SEG].",
        ]

        self.description_list = [
            "Description: {} Please answer and segment based on the above description.",
            "Definition: {} Please answer and segment based on the above definition.",
            "Description: {} Can you answer and segment it based on the above description or definition.",
            "Definition: {} Please output segmentation mask and answer based on the above description or definition.",
            "Provided description: {} Please segment accordingly.",
            "Given definition: {} Please provide segmentation and answer according to it.",
            "The description provided is: {} Now, segment it and provide your answer.",
            "Based on the provided definition: {} Please segment and provide your response.",
            "Describing the object as: {} Can you segment it accordingly?",
            "Defining it as: {} Now, segment and provide your answer.",
        ]

        self.answer_cls_list = [
            "The target is {} and the segmentation mask is [SEG].",
            "The category is {} and the mask is [SEG].",
            "It is {}, [SEG].",
            "{}, [SEG]",
            "Identified as {}, here is the segmentation: [SEG].",
            "Categorized as {}, the segmentation is: [SEG].",
            "The class is {}, and the corresponding segmentation is: [SEG].",
            "Regarding the classification, it is {}, and the segmentation is: [SEG].",
            "Classified as {}, here's the segmentation: [SEG].",
            "The label assigned is {}, and the associated segmentation is: [SEG].",
        ]

        self.answer_no_cls_list = [
            "Sorry, there is no {}",
            "No, we cannot see {}",
            "{} is not present",
            "This is {}, but not here.",
            "This is {}, however, we cannot see it.",
            "There's no sign of {} in this image.",
            "Unfortunately, {} is not visible in this image.",
            "We cannot detect {} in this image.",
            "There's no indication of {} here.",
            "Regrettably, {} cannot be observed in this image.",
        ]


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            data = self.data_list[idx]

            image_path = data['image']
            seg_path = data['mask']

            image_array = np.load(image_path) #1*32*256*256, normalized
            seg_array = np.load(seg_path)
            cls_id = int(os.path.basename(seg_path).split('_')[1].split('.')[0])

            try:
                item = {
                    'image': image_array,
                    'seg': seg_array,
                }

                it = self.transform(item)

                image = it['image']
                seg = it['seg']  # 1*D*H*W

                cls_list = self.dataset_info[self.tag]
                vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()
                if vld_cls:
                    if not self.description:
                        question_temple = random.choice(self.question_list)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.answer_list)
                    else:
                        question_temple = random.choice(self.description_list)
                        question = question_temple.format(random.choice(self.term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.answer_cls_list).format(cls_list[cls_id])
                else:
                    cls_id = random.randint(0, len(cls_list)-1)
                    if not self.description:
                        question_temple = random.choice(self.question_list)
                        question = question_temple.format(cls_list[cls_id])
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.answer_no_cls_list).format(cls_list[cls_id])
                    else:
                        question_temple = random.choice(self.description_list)
                        question = question_temple.format(random.choice(self.term_dict[cls_list[cls_id]]))
                        question = self.image_tokens + ' ' + question
                        answer = random.choice(self.answer_no_cls_list).format(cls_list[cls_id])


                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length",
                    return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[label == self.tokenizer.pad_token_id] = -100
                label[:question_len] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'seg': seg,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "seg",
                    'tag': self.tag,
                }
                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)



class RefSegDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.tokenizer = tokenizer
        self.mode = mode

        self.image_tokens = "<im_patch>" * args.proj_out_num

        self.data_list = pd.read_csv(args.refseg_data_path, engine='python')

        train_transform = mtf.Compose(
            [
                mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
                mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
                mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
                mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                mtf.ToTensord(keys=["image"], dtype=torch.float),
                mtf.ToTensord(keys=["seg"], dtype=torch.int),
            ]
        )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensord(keys=["image"], dtype=torch.float),
                    mtf.ToTensord(keys=["seg"], dtype=torch.int),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif mode == 'test':
            self.transform = val_transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                data = self.data_list.iloc[idx]
                image_path = os.path.join(self.args.data_root, data["Image"])

                image_array = np.load(image_path)  # 1*32*256*256, normalized

                seg_path = os.path.join(self.args.data_root, data["Mask"])
                seg_array = np.load(seg_path)
                seg_array = (seg_array == data["Mask_ID"]).astype(np.int8)

                item = {
                    "image": image_array,
                    "seg": seg_array,
                }

                it = self.transform(item)

                image = it['image']
                seg = it['seg']  # C*D*H*W

                question = data["Question"]
                question = self.image_tokens + ' ' + question

                answer = data["Answer"]

                self.tokenizer.padding_side = "right"
                text_tensor = self.tokenizer(
                    question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[label == self.tokenizer.pad_token_id] = -100
                label[:question_len] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'seg': seg,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    'question_type': "refseg",
                }

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, len(self.data_list) - 1)


class MultiSegDataset(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(MultiSegDataset, self).__init__()
        self.tokenizer = tokenizer

        dataset_info_path = os.path.join(args.seg_data_path, "dataset_info.json")
        with open(dataset_info_path, 'r') as f:
            self.dataset_info = json.load(f)

        self.ds_list = []
        for dataset_code in self.dataset_info.keys():
            self.ds_list.append(SegDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
            self.ds_list.append(SegDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
        self.ds_list.append(RefSegDataset(args, tokenizer, mode=mode))
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]




class PosRECTestDataset(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(PosRECTestDataset, self).__init__()
        self.tokenizer = tokenizer
        self.ds_list = []
        dataset_code = '0003'
        self.ds_list.append(PosRECDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
        self.ds_list.append(PosRECDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]



class PosREGTestDataset(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(PosREGTestDataset, self).__init__()
        self.tokenizer = tokenizer
        self.ds_list = []
        dataset_code = '0003'
        self.ds_list.append(PosREGDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
        self.ds_list.append(PosREGDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class MultiPosDataset(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(MultiPosDataset, self).__init__()
        self.tokenizer = tokenizer

        dataset_info_path = os.path.join(args.seg_data_path, "dataset_info.json")
        with open(dataset_info_path, 'r') as f:
            self.dataset_info = json.load(f)

        self.ds_list = []
        for dataset_code in self.dataset_info.keys():
            self.ds_list.append(PosRECDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
            self.ds_list.append(PosRECDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
            self.ds_list.append(PosREGDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
            self.ds_list.append(PosREGDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class UniDatasets(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(UniDatasets, self).__init__()
        self.ds_list = [
            CapDataset(args, tokenizer, mode),
            VQADataset(args, tokenizer, close_ended=True, mode=mode),
            VQADataset(args, tokenizer, close_ended=False, mode=mode),
            MultiPosDataset(args, tokenizer, mode),
            MultiSegDataset(args, tokenizer, mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]



