import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import numpy as np
import torch
from dataclasses import dataclass, field
import simple_slice_viewer as ssv
import SimpleITK as sikt
from LaMed.src.model.language_model import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(42)

@dataclass
class AllArguments:
    model_name_or_path: str = field(default="./LaMed/output/LaMed-0424-all/hf/")

    perceiver_out_num: int = field(default=256, metadata={"help": "Number of output tokens in Perceiver."})
    image_path: str = field(default="./Data/data/M3D_Cap_npy/ct_quizze/000008/Axial_C__arterial_phase.npy")


def main():
    parser = transformers.HfArgumentParser(AllArguments)
    args = parser.parse_args_into_dataclasses()[0]

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )

    # device = 'cuda' #'cpu', 'cuda'
    device = torch.device('cuda')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        image_tokens = "<im_patch>" * args.perceiver_out_num
        question = "What abnormalities can you find in CT images and what do patients need to do in their lives to prevent this abnormality from getting worse?"
        # question = "What is the largest organ in this CT image? Please output the segmentation."
        # # question = "What is the largest organ in this CT image? Please answer my question and output the bounding box."

        input = image_tokens + question

        input_id = tokenizer(input, return_tensors="pt")['input_ids'].to(device)
        image_np = np.load(args.image_path)
        image_pt = torch.from_numpy(image_np).to(device).unsqueeze(0)

        generation, seg_logit = model.generate(image_pt, input_id, seg_enable=True, max_new_tokens=256, num_beams=1, do_sample=False, temperature=1.0)
        generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
        seg_mask = (torch.sigmoid(seg_logit) > 0.5) * 1.0

        image = sikt.GetImageFromArray(image_np)
        ssv.display(image)

        seg = sikt.GetImageFromArray(seg_mask.cpu().numpy()[0])
        ssv.display(seg)

        print('question', question)
        print('generated_texts', generated_texts)

if __name__ == "__main__":
    main()
