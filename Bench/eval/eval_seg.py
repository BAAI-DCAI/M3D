import os
import csv
import random
import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from Bench.dataset.multi_dataset import SegDataset
from Bench.eval.metrics import BinaryDice
# If the model is not from huggingface but local, please uncomment and import the model architecture.
# from LaMed.src.model.language_model import *
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

    parser.add_argument('--output_dir', type=str, default="./LaMed/output/LaMed-finetune-0000/eval_seg/")
    parser.add_argument('--res', type=bool, default=False, help="RES (Referring Expression Segmentation) or SS (Segmantic Segmentation)")
    parser.add_argument('--dataset_id', type=str, default='0011', help="Which test dataset", choices=['0003', '0011', '0012'])
    parser.add_argument('--vis', type=bool, default=False)

    parser.add_argument('--seg_enable', type=bool, default=True)
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

    test_dataset = SegDataset(args, tokenizer=tokenizer, tag=args.dataset_id, description=args.res, mode='test')

    test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=32, #32
            pin_memory=True,
            shuffle=True,
            drop_last=False,
    )

    metric_fn = BinaryDice()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_path = os.path.join(args.output_dir, "eval_seg.csv")
    with open(output_path, mode='a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["idx", "Question Type", "Dataset Tag", "Question", "Answer", "Pred", "Dice"])

        for id, sample in enumerate(tqdm(test_dataloader)):
            tag = sample["tag"]
            question = sample["question"]
            question_type = sample["question_type"]
            answer = sample["answer"]
            image = sample["image"].to(device=device)
            seg = sample["seg"].to(device=device)

            input_id = tokenizer(question, return_tensors="pt")['input_ids'].to(device=device)

            with torch.inference_mode():
                generation, logits = model.generate(image, input_id, seg_enable=args.seg_enable, max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, top_p=args.top_p, temperature=args.temperature)

            generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

            pred = (torch.sigmoid(logits) > 0.5) * 1.0
            dice = metric_fn(logits, seg).item()

            # print("Answer: ", answer[0])
            # print("Prediction: ", generated_texts[0])
            # print("Dice: ", dice)

            writer.writerow([id, question_type[0], tag[0], question[0], answer[0], generated_texts[0], dice])

            if args.vis:
                desired_length = 4

                path = os.path.join(args.output_dir, 'eval_seg', str(id).zfill(desired_length))
                folder = os.path.exists(path)
                if not folder:
                    os.makedirs(path)

                batch_size, z = image.shape[0], image.shape[2]
                for b in range(batch_size):
                    with open(path + '/text.txt', 'a') as f:
                        f.write("Question: " + question[b] + "\n")
                        f.write("Answer: " + answer[b] + "\n")
                        f.write("pred: " + generated_texts[b] + "\n")

                    out = sitk.GetImageFromArray(image[b][0].detach().cpu().numpy())
                    sitk.WriteImage(out, os.path.join(path, 'image.nii.gz'))

                    out = sitk.GetImageFromArray(seg[b][0].detach().cpu().numpy())
                    sitk.WriteImage(out, os.path.join(path, 'seg.nii.gz'))

                    out = sitk.GetImageFromArray(pred[b][0].detach().cpu().numpy())
                    sitk.WriteImage(out, os.path.join(path, 'pred.nii.gz'))

if __name__ == "__main__":
    main()
       