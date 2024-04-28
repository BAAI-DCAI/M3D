# M3D: Advancing 3D Medical Image Analysis with Multi-Modal Large Language Models

<font size=3><div align='center' > <a href=https://arxiv.org/abs/2404.00578>**Paper**</a> | [**Data**](#M3D-Data) | [**Model**](#model) | [**Training**](#training) | [**Inference**](#inference) | [**Benchmark**](#benchmark) | [**Online Demo**]()</div></font>
M3D is the pioneering and comprehensive series of work on the  multi-modal large language model for 3D medical analysis, including:
- **M3D-Data**: the largest-scale open-source 3D medical dataset, consists of 120K image-text pairs and 662K instruction-response pairs;
- **M3D-LaMed**: the versatile multi-modal models with M3D-CLIP pretrained vision encoder, which are capable of tasks such as image-text retrieval, report generation, visual question answering, positioning and segmentation;
- **M3D-Bench**: the most comprehensive automatic evaluation benchmark covers 8 tasks.


### News
- [x] [2024.04.28] We have released the data, code and model, and we will improve the README as soon as possible.


### Quickstart


### Installation

### Data Preparation

### Training
#### Pretrained Weights
#### Training
#### Merge LoRA Weight

### Benchmark
#### Evaluation


### Model
| Model    | Download Link                                                                                                                                 |
|----------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| M3D-CLIP | [HuggingFace](https://huggingface.co/GoodBaiBai88/M3D-CLIP), [ModelScope]()    |
| M3D-LaMed-Llama-2-7B  | [HuggingFace](https://huggingface.co/GoodBaiBai88/M3D-LaMed-Llama-2-7B), [ModelScope]()|


### M3D-Data
| Dataset  | Type | Images | Texts | Download Link |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| M3D-Cap | 3D image-text pairs |	120,092 | 42,496 | [HuggingFace](https://huggingface.co/datasets/GoodBaiBai88/M3D-Cap), [ModelScope](https://www.modelscope.cn/datasets/GoodBaiBai88/M3D-Cap) |
| M3D-VQA | 3D images, questions, and answers |	96,170 | 509,755 | [HuggingFace](https://huggingface.co/datasets/GoodBaiBai88/M3D-VQA), [ModelScope](https://www.modelscope.cn/datasets/GoodBaiBai88/M3D-VQA) |
| M3D-Seg | 3D images, category text, and segmentation masks | 5,772 | 149,196 | [HuggingFace](https://huggingface.co/datasets/GoodBaiBai88/M3D-Seg), [ModelScope](https://www.modelscope.cn/datasets/GoodBaiBai88/M3D-Seg) |
| M3D-RefSeg | 3D images, questions, answers, and segmentation masks |	210 | 2,778 | [HuggingFace](https://huggingface.co/datasets/GoodBaiBai88/M3D-RefSeg), [ModelScope](https://www.modelscope.cn/datasets/GoodBaiBai88/M3D-RefSeg) |

### Citation

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

### Acknowledgement
