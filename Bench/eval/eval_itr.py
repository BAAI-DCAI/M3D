import os
import csv
import random
from torch.utils.data import DataLoader
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
from Bench.dataset.multi_dataset import ITRDataset
from LaMed.src.model.CLIP import *


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
    parser.add_argument('--model_name_or_path', type=str, default="GoodBaiBai88/M3D-CLIP", choices=[])
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])

    # data
    parser.add_argument('--data_root', type=str, default="./Data/data/")
    parser.add_argument('--cap_data_path', type=str, default="./Data/data/M3D_Cap_npy/M3D_Cap_eh.json")
    parser.add_argument('--output_dir', type=str, default="./LaMed/output/CLIP/eval_itr/")
    parser.add_argument('--save_output', type=bool, default=False)

    return parser.parse_args(args)



def calculate_recall(similarity_matrix, k):
    _, topk_indices = similarity_matrix.topk(k, dim=1)
    diagonal_indices = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
    correct_matches = torch.eq(topk_indices, diagonal_indices.view(-1, 1))
    recall_at_k = correct_matches.float().sum(dim=1).mean()
    return recall_at_k


def main():
    seed_everything(42)
    args = parse_args()
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.max_length,
        padding_side="right",
        use_fast=False
    )
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    model = model.to(device=device)

    test_dataset = ITRDataset(args, tokenizer=tokenizer, mode='hard_test') # test, test1k, test500, test100

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

    txt_feats_all = []
    img_feats_all = []
    for sample in tqdm(test_dataloader):
        input_id = sample["input_id"].to(device=device)
        attention_mask = sample["attention_mask"].to(device=device)
        image = sample["image"].to(device=device)
        with torch.inference_mode():
            image_features = model.encode_image(image)[:, 0]
            text_features = model.encode_text(input_id, attention_mask)[:, 0]
        txt_feats_all.append(text_features.detach().cpu())
        img_feats_all.append(image_features.detach().cpu())

    txt_feats_all = torch.cat(txt_feats_all, dim=0)
    img_feats_all = torch.cat(img_feats_all, dim=0)

    scores_mat = torch.matmul(img_feats_all, txt_feats_all.transpose(0, 1))

    ir_r1 = calculate_recall(scores_mat, 1)
    ir_r5 = calculate_recall(scores_mat, 5)
    ir_r10 = calculate_recall(scores_mat, 10)
    tr_r1 = calculate_recall(scores_mat.transpose(0, 1), 1)
    tr_r5 = calculate_recall(scores_mat.transpose(0, 1), 5)
    tr_r10 = calculate_recall(scores_mat.transpose(0, 1), 10)

    print("IR_R1: ", ir_r1, "\n",
          "IR_R5: ", ir_r5, "\n",
          "IR_R10: ", ir_r10, "\n",
          "TR_R1: ", tr_r1, "\n",
          "TR_R5: ", tr_r5, "\n",
          "TR_R10: ", tr_r10, "\n")

    if args.save_output:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        output_path = os.path.join(args.output_dir, "test_ir.csv")
        with open(output_path, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["id", "top1", "top10"])
            for i in tqdm.tqdm(range(len(scores_mat))):
                scores_vect = scores_mat[i]
                top_values, top_indices = torch.topk(scores_vect, k=10)
                max_index = torch.argmax(scores_vect).item()
                top_id_list = top_indices.cpu().detach().tolist()
                writer.writerow([i, max_index, top_id_list])

        output_path = os.path.join(args.output_dir, "test_tr.csv")
        with open(output_path, mode='w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["id","top1", "top10"])
            for i in tqdm.tqdm(range(len(scores_mat))):
                scores_vect = scores_mat.transpose(0, 1)[i]
                top_values, top_indices = torch.topk(scores_vect, k=10)
                max_index = torch.argmax(scores_vect).item()
                top_id_list = top_indices.cpu().detach().tolist()
                writer.writerow([i, max_index, top_id_list])
        print("Save test csv successfully!")

if __name__ == "__main__":
    main()
       