import argparse
import os
import re
import sys
import bleach
import gradio as gr
import numpy as np
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import nibabel as nib
import numpy as np
from PIL import Image
from monai.transforms import Resize
from LaMed.src.model.language_model import *


def parse_args(args):
    parser = argparse.ArgumentParser(description="M3D-LaMed chat")
    parser.add_argument('--model_name_or_path', type=str, default="./LaMed/output/LaMed-Phi3-4B-finetune-0000/hf/", choices=[])
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--device', type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument('--seg_enable', type=bool, default=True)
    parser.add_argument('--proj_out_num', type=int, default=256)

    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def image_process(file_path):
    if file_path.endswith('.nii.gz'):
        nifti_img = nib.load(file_path)
        img_array = nifti_img.get_fdata()
    elif file_path.endswith(('.png', '.jpg', '.bmp')):
        img = Image.open(file_path).convert('L')
        img_array = np.array(img)
        img_array = img_array[np.newaxis, :, :]
    elif file_path.endswith('.npy'):
        img_array = np.load(file_path)
    else:
        raise ValueError("Unsupported file type")

    resize = Resize(spatial_size=(32, 256, 256), mode="bilinear")
    img_meta = resize(img_array)
    img_array, img_affine = img_meta.array, img_meta.affine

    return img_array, img_affine

args = parse_args(sys.argv[1:])
os.makedirs(args.vis_save_path, exist_ok=True)

# Create model
device = torch.device(args.device)

dtype = torch.float32
if args.precision == "bf16":
    dtype = torch.bfloat16
elif args.precision == "fp16":
    dtype = torch.half

kwargs = {"torch_dtype": dtype}
if args.load_in_4bit:
    kwargs.update(
        {
            "torch_dtype": torch.half,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        }
    )
elif args.load_in_8bit:
    kwargs.update(
        {
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                llm_int8_skip_modules=["visual_model"],
                load_in_8bit=True,
            ),
        }
    )


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
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    **kwargs
)
model = model.to(device=device)

model.eval()

# Gradio
examples = [
    [
        "/mnt/hpfs/baaidcai/baifan/M3D/Data/data/examples/example_00.npy",
        "Please generate a medical report based on this image.",
    ],
    [
        "/mnt/hpfs/baaidcai/baifan/M3D/Data/data/examples/example_01.npy",
        "What is the abnormality type in this image?",
    ],
    [
        "/mnt/hpfs/baaidcai/baifan/M3D/Data/data/examples/example_02.npy",
        "What is the plane of the CT image? Choices: A. Axial B. Sagittal C. Coronal D. Oblique",
    ],
    [
        "/mnt/hpfs/baaidcai/baifan/M3D/Data/data/examples/example_03.npy",
        "Where is liver in this image? Please output the box.",
    ],
    [
        "/mnt/hpfs/baaidcai/baifan/M3D/Data/data/examples/example_04.npy",
        "Can you segment the lung in this image? Please output the mask.",
    ],
    [
        "/mnt/hpfs/baaidcai/baifan/M3D/Data/data/examples/example_05.npy",
        "Can you find the organs related to the balance of water and salt? Please output the box.",
    ],
]

description = """
Due to resource limitations, we run the half-precision (bfloat16) M3D-LaMed-Llama2-7B model on NVIDIA RTX 3090 24G for online demo. \n
You can try better model and performance on our full-precision model in HuggingFace. \n
If multiple users are using at the same time, there may delay some time. \n

**Note**: Different prompts can lead to significantly varied results. \n
**Note**: Please try to **standardize** your input text prompts to **avoid ambiguity**, and also pay attention to whether the **punctuations** of the input are correct. \n
**Note**: Current model is **M3D-LaMed-Llama2-7B**, and half-precision may impair performance. \n

**Usage**: <br>
&ensp;(1) Report Generation, input prompt like : "Please generate a medical report based on this image."; <br>
&ensp;(2) Open-ended VQA, input prompt like : "Is/Which/What xxx in this image?"; <br>
&ensp;(3) Closed-ended VQA, input prompt like : "Which/What xxx in this image? Choices: A. xxx B. xxx C. xxx D. xxx"; <br>
&ensp;(4) Positioning, input prompt like : "Where is xxx in this image? Please output the box."; <br>
&ensp;(5) Segmentation, input prompt like : "Can you segment the xxx in this image? Please output the mask."; <br>

We found that M3D-LaMed has generalization and reasoning capabilities, and we can try some diverse and interesting problems, such as: \n
&ensp;(1) "Can you find the organ related to breathing? Please output the mask."; <br>
&ensp;(2) "What is the largest organ in the image? Please output the box."; <br>

Although M3D-LaMed still has certain shortcomings, we believe that our model will become better with the addition of more data and researchers. \n 
If you recognize our work or think our work is helpful, please support us through [🌟Github Star](https://github.com/BAAI-DCAI/M3D)
Hope you can enjoy our work!
"""

title_markdown = ("""
# M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models

[📖[Paper](https://arxiv.org/abs/2404.00578)] | [🏠[Code](https://github.com/BAAI-DCAI/M3D)] | [🤗[Model](https://huggingface.co/GoodBaiBai88/M3D-LaMed-Llama-2-7B)]
""")


def extract_box_from_text(text):
    match = re.search(r'\[([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+), ([-+]?\d*\.\d+|[-+]?\d+)\]', text)
    if match:
        box_coordinates = [float(coord) for coord in match.groups()]
        return box_coordinates
    else:
        return None

## to be implemented
def inference(input_image, input_str, temperature, top_p):
    global vis_box
    global seg_mask
    vis_box = [0, 0, 0, 0, 0, 0]
    seg_mask = np.zeros((32, 256, 256), dtype=np.uint8)

    ## filter out special chars
    input_str = bleach.clean(input_str)

    print("input_str: ", input_str, "input_image: ", input_image)

    # Model Inference
    prompt = "<im_patch>" * args.proj_out_num + input_str

    input_id = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device=device)
    image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)

    generation, seg_logit = model.generate(image_pt, input_id, seg_enable=args.seg_enable, max_new_tokens=args.max_new_tokens,
                                        do_sample=args.do_sample, top_p=top_p, temperature=temperature)

    output_str = tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
    print("output_str", output_str)
    box = extract_box_from_text(output_str)
    if box is not None:
        vis_box = [box[0]*32, box[1]*256, box[2]*256, box[3]*32, box[4]*256, box[5]*256]
        vis_box = [int(b) for b in vis_box]
        return output_str, (image_rgb[0], [((0,0,0,0), 'target')])

    seg_mask = (torch.sigmoid(seg_logit) > 0.5).squeeze().detach().cpu().numpy()
    if seg_mask.sum() == 0:
        return output_str, None
    else:
        return output_str, (image_rgb[0], [(seg_mask[0], 'target')])


def select_slice(selected_slice):
    min_s = min(vis_box[0], vis_box[3])
    max_s = max(vis_box[0], vis_box[3])

    if min_s <= selected_slice <= max_s:
        return (image_rgb[selected_slice], [(seg_mask[selected_slice], 'target_mask'), ((vis_box[2],vis_box[1], vis_box[5],vis_box[4]), 'target_box')])
    else:
        return (image_rgb[selected_slice], [(seg_mask[selected_slice], 'target_mask'), ((0,0,0,0), 'target_box')])


def load_image(load_image):
    global image_np
    global image_rgb
    global vis_box
    global seg_mask
    vis_box = [0, 0, 0, 0, 0, 0]
    seg_mask = np.zeros((32, 256, 256), dtype=np.uint8)

    image_np, image_affine = image_process(load_image)
    image_rgb = (np.stack((image_np[0],) * 3, axis=-1) * 255).astype(np.uint8)

    return (image_rgb[0], [((0,0,0,0), 'target_box')])


with gr.Blocks() as demo:
    gr.Markdown(title_markdown)
    gr.Markdown(description)
    with gr.Row():
        with gr.Column():
            image = gr.File(type="filepath", label="Input File")
            text = gr.Textbox(lines=1, placeholder=None, label="Text Instruction")
            with gr.Accordion("Parameters", open=False) as parameter_row:
                temperature = gr.Slider(minimum=0.0, maximum=1.0, value=0.2, step=0.1, interactive=True,
                                        label="Temperature", )
                top_p = gr.Slider(minimum=0.0, maximum=1.0, value=0.7, step=0.1, interactive=True, label="Top P", )
            with gr.Row():
                btn_c = gr.ClearButton([image, text])
                btn = gr.Button("Run")
            text_out = gr.Textbox(lines=1, placeholder=None, label="Text Output")
        with gr.Column():
            image_out = gr.AnnotatedImage(color_map={"target_mask": "#a89a00", "target_box": "#ffae00"})
            slice_slider = gr.Slider(minimum=0, maximum=31, step=1, interactive=True, scale=1, label="Selected Slice")

    gr.Examples(examples=examples, inputs=[image, text])

    image.change(fn=load_image, inputs=[image], outputs=[image_out])
    btn.click(fn=inference, inputs=[image, text, temperature, top_p], outputs=[text_out, image_out])
    slice_slider.change(fn=select_slice, inputs=slice_slider, outputs=[image_out])
    btn_c.click()

demo.queue()
demo.launch(share=True)


