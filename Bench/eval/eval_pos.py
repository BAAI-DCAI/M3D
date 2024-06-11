import os
import csv
import random
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from Bench.dataset.multi_dataset import PosRECTestDataset, PosREGTestDataset
from Bench.utils import extract_box_from_text, calculate_iou
# If the model is not from huggingface but local, please uncomment and import the model architecture.
# from LaMed.src.model.language_model import *
import matplotlib.pyplot as plt
import SimpleITK as sitk
import evaluate

accuracy = evaluate.load("accuracy")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="GoodBaiBai88/M3D-LaMed-Llama-2-7B", choices=[])
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])

    # data
    parser.add_argument('--data_root', type=str, default="./Data/data")
    parser.add_argument('--seg_data_path', type=str, default="./Data/data/M3D_Seg_npy/")
    parser.add_argument('--term_dict_path', type=str, default="./Data/data/M3D_Seg_npy/term_dictionary.json")

    parser.add_argument('--rec', type=bool, default=True, help="REC with box output")
    parser.add_argument('--output_dir', type=str, default="./LaMed/output/LaMed-finetune-0000/eval_pos/")
    parser.add_argument('--vis', type=bool, default=False)

    parser.add_argument('--proj_out_num', type=int, default=256)

    return parser.parse_args(args)

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


          
def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map='auto',
        trust_remote_code=True
    )
    model = model.to(device=device)

    if args.rec:
        test_dataset = PosRECTestDataset(args, tokenizer, mode='test')
    else:
        test_dataset = PosREGTestDataset(args, tokenizer, mode='test')

    test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=32,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
    )

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.rec:
        output_path = os.path.join(args.output_dir, "eval_rec.csv")
        with open(output_path, mode='w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["idx", "Question Type", "Question", "Answer", "pred", "IOU"])
            for id, sample in enumerate(tqdm(test_dataloader)):
                question = sample["question"]
                question_type = sample["question_type"]
                answer = sample['answer']

                image = sample["image"].to(device=device)
                input_id = tokenizer(question, return_tensors="pt")['input_ids'].to(device=device)

                with torch.inference_mode():
                    generation = model.generate(image, input_id, max_new_tokens=args.max_new_tokens,
                                                do_sample=args.do_sample, top_p=args.top_p,
                                                temperature=args.temperature)
                generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

                question_box = extract_box_from_text(question[0])
                prediction_box = extract_box_from_text(generated_texts[0])
                answer_box = extract_box_from_text(answer[0])

                if prediction_box and answer_box:
                    iou = calculate_iou(prediction_box, answer_box)
                else:
                    iou = -1

                print("Answer: ", answer[0])
                print("Pred: ", generated_texts[0])
                print("IOU: ", iou)

                writer.writerow([id, question_type, question[0], answer[0], generated_texts[0], iou])

                if args.vis:
                    desired_length = 4
                    path = os.path.join(args.output_dir, 'eval_pos', str(id).zfill(desired_length))
                    folder = os.path.exists(path)
                    if not folder:
                        os.makedirs(path)

                    # image draw
                    image = image[0][0]
                    slice = image.shape[0]

                    out = sitk.GetImageFromArray(image.detach().cpu().numpy())
                    sitk.WriteImage(out, os.path.join(path, 'image.nii.gz'))

                    for i in range(slice):
                        image_slice = image[i]
                        plt.imshow(image_slice.cpu().numpy(), cmap='gray')

                        if answer_box:
                            x1, y1, z1, x2, y2, z2 = answer_box
                            d1, h1, w1, d2, h2, w2 = int(x1*32), int(y1*256), int(z1*256), int(x2*32), int(y2*256), int(z2*256)
                            if i >= d1 and i <= d2:
                                plt.plot([w1, w1, w2, w2, w1], [h1, h2, h2, h1, h1], color='darkgreen', linewidth=5)
                        if question_box:
                            x1, y1, z1, x2, y2, z2 = question_box
                            d1, h1, w1, d2, h2, w2 = int(x1*32), int(y1*256), int(z1*256), int(x2*32), int(y2*256), int(z2*256)
                            if i >= d1 and i <= d2:
                                plt.plot([w1, w1, w2, w2, w1], [h1, h2, h2, h1, h1], color='darkgreen', linewidth=5)
                        if prediction_box:
                            x1, y1, z1, x2, y2, z2 = prediction_box
                            d1, h1, w1, d2, h2, w2 = int(x1*32), int(y1*256), int(z1*256), int(x2*32), int(y2*256), int(z2*256)
                            if i >= d1 and i <= d2:
                                plt.plot([w1, w1, w2, w2, w1], [h1, h2, h2, h1, h1], color='darkred', linewidth=5)

                        plt.axis('off')
                        plt.savefig(os.path.join(path, str(i).zfill(desired_length) + 'image.png'), bbox_inches='tight', pad_inches=0)
                        plt.close()
    else:
        output_path = os.path.join(args.output_dir, "eval_reg.csv")
        with open(output_path, mode='w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["idx", "Question", "Answer", "Pred", "bleu", "rouge1", "meteor", "bert_f1"])
            for id, sample in enumerate(tqdm.tqdm(test_dataloader)):
                question = sample["question"]
                answer = sample['answer']

                image = sample["image"].to(device=device)
                input_id = tokenizer(question, return_tensors="pt")['input_ids'].to(device=device)

                with torch.inference_mode():
                    generation = model.generate(image, input_id, max_new_tokens=args.max_new_tokens,
                                                do_sample=args.do_sample, top_p=args.top_p,
                                                temperature=args.temperature)
                generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

                result = dict()
                decoded_preds, decoded_labels = postprocess_text(generated_texts, answer)
                bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
                result["bleu"] = bleu_score['bleu']

                rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=['rouge1'])
                result["rouge1"] = rouge_score['rouge1']

                meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
                result["meteor"] = meteor_score['meteor']

                bert_score = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
                result["bert_f1"] = sum(bert_score['f1']) / len(bert_score['f1'])

                writer.writerow(
                    [id, question[0], answer[0], generated_texts[0], result["bleu"], result["rouge1"], result["meteor"], result["bert_f1"]])


if __name__ == "__main__":
    main()
       