# M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models

[demo]:https://github.com/BAAI-DCAI/M3D/blob/main/LaMed/src/demo/online_demo.py

<font size=3><div align='center' > <a href=https://arxiv.org/abs/2404.00578>**Paper**</a> | [**Data**](#data) | [**Model**](#model) | [**Training**](#training) | [**Benchmark**](#benchmark) | [**Online Demo**][demo]</div></font>
M3D is the pioneering and comprehensive series of work on the  multi-modal large language model for 3D medical analysis, including:
- **M3D-Data**: the largest-scale open-source 3D medical dataset, consists of 120K image-text pairs and 662K instruction-response pairs;
- **M3D-LaMed**: the versatile multi-modal models with M3D-CLIP pretrained vision encoder, which are capable of tasks such as image-text retrieval, report generation, visual question answering, positioning and segmentation;
- **M3D-Bench**: the most comprehensive automatic evaluation benchmark covers 8 tasks.

## Notifications
📢 [2024.06.12]
- 🔥🔥🔥 We released an [online demo][demo]. Welcome everyone to try it now!
- We released a light but strong model, M3D-LaMed-Phi-3-4B. After simple testing, it outperformed M3D-LaMed-Llama-2-7B. We are conducting detailed experiments. Please try it first.
- We found that the previous M3D-LaMed-Llama-2-7B model had problems in the segmentation task. We have fixed this problem and will re-release the new model in the next few days.

## News
- [x] [2024.06.14] We released a light but strong model [M3D-LaMed-Phi-3-4B](https://huggingface.co/GoodBaiBai88/M3D-LaMed-Phi-3-4B) and an [online demo][demo].
- [x] [2024.04.28] We released the data, code, and model.


## Quickstart
Here, we can easily use our model based on Hugging Face.

```python
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import simple_slice_viewer as ssv
import SimpleITK as sikt

device = torch.device('cuda') # 'cpu', 'cuda'
dtype = torch.bfloat16 # or bfloat16, float16, float32

model_name_or_path = 'GoodBaiBai88/M3D-LaMed-Phi-3-4B'
proj_out_num = 256

# Prepare your 3D medical image:
# 1. The image shape needs to be processed as 1*32*256*256, consider resize and other methods.
# 2. The image needs to be normalized to 0-1, consider Min-Max Normalization.
# 3. The image format needs to be converted to .npy 
# 4. Although we did not train on 2D images, in theory, the 2D image can be interpolated to the shape of 1*32*256*256 for input.
image_path = "./Data/data/examples/example_03.npy"

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=dtype,
    device_map='auto',
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    model_max_length=512,
    padding_side="right",
    use_fast=False,
    trust_remote_code=True
)

model = model.to(device=device)

# question = "Can you provide a caption consists of findings for this medical image?"
question = "What is liver in this image? Please output the segmentation mask."
# question = "What is liver in this image? Please output the box."

image_tokens = "<im_patch>" * proj_out_num
input_txt = image_tokens + question
input_id = tokenizer(input_txt, return_tensors="pt")['input_ids'].to(device=device)

image_np = np.load(image_path)
image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)

# generation = model.generate(image_pt, input_id, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)
generation, seg_logit = model.generate(image_pt, input_id, seg_enable=True, max_new_tokens=256, do_sample=True, top_p=0.9, temperature=1.0)

generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)
seg_mask = (torch.sigmoid(seg_logit) > 0.5) * 1.0

print('question', question)
print('generated_texts', generated_texts[0])

image = sikt.GetImageFromArray(image_np)
ssv.display(image)
seg = sikt.GetImageFromArray(seg_mask.cpu().numpy()[0])
ssv.display(seg)
```

## Model
| Model    | Download Link                                                                                                                                 |
|----------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| M3D-CLIP | [HuggingFace](https://huggingface.co/GoodBaiBai88/M3D-CLIP), [ModelScope]()    |
| M3D-LaMed-Phi-3-4B  | [HuggingFace](https://huggingface.co/GoodBaiBai88/M3D-LaMed-Phi-3-4B), [ModelScope]()|
| M3D-LaMed-Llama-2-7B  | [HuggingFace](https://huggingface.co/GoodBaiBai88/M3D-LaMed-Llama-2-7B), [ModelScope]()|


## Installation
```bash
git clone https://github.com/BAAI-DCAI/M3D.git
pip install -r requirements.txt
```

## Data
M3D-Data supports the training and benchmark, which consist of 4 types of data:

| Dataset  | Type | Images | Texts | Download Link |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| M3D-Cap | 3D image-text pairs |	120,092 | 42,496 | [HuggingFace](https://huggingface.co/datasets/GoodBaiBai88/M3D-Cap), [ModelScope](https://www.modelscope.cn/datasets/GoodBaiBai88/M3D-Cap) |
| M3D-VQA | 3D images, questions, and answers |	96,170 | 509,755 | [HuggingFace](https://huggingface.co/datasets/GoodBaiBai88/M3D-VQA), [ModelScope](https://www.modelscope.cn/datasets/GoodBaiBai88/M3D-VQA) |
| M3D-Seg | 3D images, category text, and segmentation masks | 5,772 | 149,196 | [HuggingFace](https://huggingface.co/datasets/GoodBaiBai88/M3D-Seg), [ModelScope](https://www.modelscope.cn/datasets/GoodBaiBai88/M3D-Seg) |
| M3D-RefSeg | 3D images, questions, answers, and segmentation masks |	210 | 2,778 | [HuggingFace](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg), [ModelScope](https://www.modelscope.cn/datasets/GoodBaiBai88/M3D-RefSeg) |

Please follow the instructions for each dataset to download and preprocess.
You can find the preprocess file named `m3d_xx_data_prepare.py` in dataset description or `Data/process/` for M3D-Cap, M3D-Seg and M3D-RefSeg.
We recommend saving the downloaded and processed dataset to `Data/data/`.

## Training
### Pretrained Weights
To train M3D-LaMed, you need to prepare some pretrained weights for better performance and faster convergence.

#### Vision encoder
We recommend downloading the medical 3D ViT weight `pretrained_ViT.bin` from [M3D-CLIP](https://huggingface.co/GoodBaiBai88/M3D-CLIP/tree/main) directly.
Or you can also pretrain the 3D ViT by yourself by
```bash
sh LaMed/script/train_clip.sh
```

#### LLM
Phi-3-4B: Download and follow [here](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct).
Llama-2-7B: Download and follow [here](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

#### Segmentation module
SegVol: Download and follow [here](https://huggingface.co/BAAI/SegVol/tree/main).

### Training
Our training consists of two steps. 
- **Step 1**: [Pretrain](#step-1-pretrain)
- **Step 2**: [Visual Instruction Tuning](#step-2-visual-instruction-tuning)

#### Configuration
We suggest using `accelerate` to train. It was developed by Hugging Face 
and conveniently supports common training strategies such as distributed training, mixed precision, DeepSpeed, etc.
It should be configured on first use:
```bash
accelerate config
```
Please follow the configuration guide and we can choose the appropriate training strategy. 
We recommend using bf16 and Deepspeed for acceleration, and the ZeRO type depends on your own situation.

If you don't know how to configure it, we provide a simple configuration `default_config.yaml` for your reference.
<details>
<summary>default_config.yaml</summary>

```bash
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 1
  zero3_init_flag: false
  zero_stage: 0
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
</details>

#### Step 1: Pretrain
We should align vision and language with image-text data, that is, only train mm_projector and freeze the vision encoder and LLM. 

Please update LLM path `--model_name_or_path` and vision encoder path `--pretrain_vision_model`, respectively.
Update `--output_dir` to specify the output path of the model.
Then run the script by:
```bash
sh LaMed/script/pretrain_phi3.sh
```

#### Step 2: Visual Instruction Tuning
Visual instruction tuning through multi-task data of image-text pairs, VQA, positioning and segmentation, 
that is, only perform LoRA training on LLM, and unfreeze all other models.

Please update LLM path `--model_name_or_path`, vision encoder path `--pretrain_vision_model`, model path saved by Step 1`--pretrain_mm_mlp_adapter` and segmentation module path `--pretrain_seg_module`, respectively.
Update `--output_dir` to specify the output path of the model.
Then run the script by:
```bash
sh LaMed/script/finetune_lora_phi3.sh
```

### Merge LoRA Weight
Merge the LoRA weights of `model_with_lora.bin`, save the final model into your desired path in the Hugging Face format:
```bash
CUDA_VISIBLE_DEVICES="" python merge_lora_weights_and_save_hf_model.py \
  --version="" \
  --model_type="" \
  --model_with_lora="PATH_TO_model_with_lora.bin" \
  --output_dir="PATH_TO_SAVED_MODEL"
```

## Benchmark
We propose the most comprehensive automatic evaluation benchmark covers 8 tasks in 3D medical, including 
image-text retrival, report generation, closed-ended VQA, open-ended VQA, referring expression comprehension,
referring expression generation, semantic segmentation, referring expression segmentation.

### Evaluation
We can directly evaluate each task by running:
```bash
CUDA_VISIBLE_DEVICES="" python Bench/eval/eval_TASK.py
```

We also provide a more accurate automatic evaluation of report generation tasks using LLM, 
after modifying the `file_path`, please run:
```bash
CUDA_VISIBLE_DEVICES="" python Bench/eval/eval_with_llm.py
```

## Dataset Copyright Information
All images and reports involved in this dataset are publicly available data. The M3D-Data have obtained an official license approval from Radiopaedia. We support the non-commercial use of Radiopaedia content for machine learning.

## Citation
If our dataset or project are helpful to you, please consider citing:

```BibTeX
@misc{bai2024m3d,
      title={M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models}, 
      author={Fan Bai and Yuxin Du and Tiejun Huang and Max Q. -H. Meng and Bo Zhao},
      year={2024},
      eprint={2404.00578},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@article{du2023segvol,
  title={SegVol: Universal and Interactive Volumetric Medical Image Segmentation},
  author={Du, Yuxin and Bai, Fan and Huang, Tiejun and Zhao, Bo},
  journal={arXiv preprint arXiv:2311.13385},
  year={2023}
}
```

## Acknowledgement
We appreciate open source projects including: 
[LLaVA](https://github.com/haotian-liu/LLaVA), 
[LISA](https://github.com/dvlab-research/LISA), 
[SegVol](https://github.com/BAAI-DCAI/SegVol).
