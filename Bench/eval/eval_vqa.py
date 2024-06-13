import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from Bench.dataset.multi_dataset import VQADataset
from Bench.eval.metrics import compute_exact_match, qa_f1_score
# If the model is not from huggingface but local, please uncomment and import the model architecture.
# from LaMed.src.model.language_model import *
import evaluate

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
    parser.add_argument('--vqa_data_test_path', type=str, default="./Data/data/M3D-VQA/M3D_VQA_test.csv")
    parser.add_argument('--close_ended', type=bool, default=True)
    parser.add_argument('--output_dir', type=str, default="./LaMed/output/LaMed-Phi3-4B-finetune-0000/eval_vqa/")

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

    test_dataset = VQADataset(args, tokenizer=tokenizer, close_ended=args.close_ended, mode='test')

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

    if args.close_ended:
        output_path = os.path.join(args.output_dir, "eval_close_vqa.csv")
        with open(output_path, mode='w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["Question Type", "Question", "Answer", "Answer Choice", "Pred", "Correct"])
            for sample in tqdm(test_dataloader):
                question = sample["question"]
                question_type = sample["question_type"].item()
                answer_choice = sample["answer_choice"]
                answer = sample['answer']

                image = sample["image"].to(device=device)

                input_id = tokenizer(question, return_tensors="pt")['input_ids'].to(device=device)

                with torch.inference_mode():
                    generation = model.generate(image, input_id, max_new_tokens=args.max_new_tokens,
                                                do_sample=args.do_sample, top_p=args.top_p,
                                                temperature=args.temperature)
                generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

                if answer_choice[0] + '.' in generated_texts[0]:
                    correct = 1
                else:
                    correct = 0

                writer.writerow([question_type, question[0], answer[0], answer_choice[0], generated_texts[0], correct])
    else:
        output_path = os.path.join(args.output_dir, "eval_open_vqa.csv")
        with open(output_path, mode='w') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["Question Type", "Question", "Answer", "Pred", "bleu", "rouge1", "meteor", "bert_f1"])
            for sample in tqdm(test_dataloader):
                question = sample["question"]
                question_type = sample["question_type"].item()
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
                    [question_type, question[0], answer[0], generated_texts[0], result["bleu"], result["rouge1"], result["meteor"], result["bert_f1"]])

if __name__ == "__main__":
    main()
       