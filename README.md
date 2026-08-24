# [ICCV'25] OURO: A Self-Bootstrapped Framework for Multimodal Scene Understanding

## Overview
OURO is a self-bootstrapped framework designed to enhance multimodal scene understanding by generating multi-level Region Proposal Networks (RPN) and hierarchical scene descriptions. This project provides implementations of various components, including multi-level RPN generation, captioning, visual question answering (VQA), and a training pipeline for fine-tuned multimodal models.

## Repository Structure

- **`/multi_level_rpn_description_generation.py`**: The script for generating multi-level RPN and hierarchical descriptions. This code requires the `detectron2` environment to function correctly, and the setup instructions are linked below.

- **`/generated_data/`**: Placeholder directory for generated captions and VQA pairs (TBD—please note that due to the size of the generated data, it may not be available immediately).

- **`/web_demo/`**: Web-based demonstration (coming soon).

- **`/training/`**: Instructions and setup for training OURO with custom datasets. This section uses the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) repository for dataset creation and model training, with the flexibility to choose sub-regions for training.

## Requirements

### 1. Environment Setup (Detectron2)
The multi-level RPN generation script requires **Detectron2**. Follow the instructions from the official [Detectron2 installation guide](https://github.com/facebookresearch/detectron2/blob/main/tools/install.md) to set up the environment.

Here’s a quick installation guide:

1. Clone the Our repository:
    ```bash
    git clone https://github.com/tinnel123666888/OURO.git
    cd detectron2-main
    ```

2. Install dependencies:
    ```bash
    python -m pip install -e .
    ```

3. Verify the installation:
    ```bash
    python -c "import detectron2"
    ```
    

Once you have Detectron2 set up, you can run the multi-level RPN generation code. If you want to generate your own dataset, you can follow the datasets mentioned in the [Monkey repository](https://github.com/Yuliang-Liu/Monkey) to create a multi-level RPN dataset.

4. Run the multi-level RPN generation script:
   To generate multi-level RPN and associated descriptions, run the following command:

   ```bash
   python multi_level_rpn_description_generation.py \
    --base_image_dir /path/to/your/images \
    --base_output_dir /path/to/output/images \
    --use_cuda \
    --message_file /path/to/save/messages.json \
    --confidence_threshold 0.35
   ```

--base_image_dir: Path to the root directory containing all images.

--base_output_dir: Directory to save the processed output images.

--use_cuda: Enable CUDA for GPU processing (optional).

--message_file: Path where the messages.json will be saved.

--confidence_threshold: Confidence threshold for RPN object detection (default is 0.35).

### 2. Caption and VQA Generation 

The VQA dataset is available at https://huggingface.co/datasets/tinnel123/OURO_dataset

If you wish to generate the dataset on your own, you can follow the procedure outlined below after performing multi-level RPN generation and using **Qwen2-VL** for inference. After generating the RPNs and using Qwen2-VL to perform inference, the following steps are used to generate captions and VQA pairs:

1. **Merge Hierarchical Captions**: After generating captions for different hierarchical levels, these captions are merged by matching key terms between parent and child captions. This ensures a coherent and comprehensive description that spans all levels.
   
**Prepare Your Data Folder Structure and JSON File**

To enable hierarchical caption merging, **organize your dataset as follows**:

* Place your images in a nested folder structure that reflects the hierarchy you want to capture.
  For example:

  ```
  /your_dataset_root/
    root_image1/
        data.json
        root_img.jpg
        child1/
            child1_img.jpg
            grandchild1/
                grandchild1_img.jpg
        child2/
            child2_img.jpg
            grandchild2/
                grandchild2_img.jpg
    root_image2/
        data.json
        root_img.jpg
        child1/
            child1_img.jpg
            grandchild1/
                grandchild1_img.jpg
        child2/
            child2_img.jpg
            grandchild2/
                grandchild2_img.jpg

  ```

* After running multi-level RPN and Qwen2-VL inference, save the resulting captions for each image in a JSON file (`your_json.json`) with the following structure:

  ```json
  [
    [
      [
        {
          "image_path": "/your_dataset_root/root/root_img.jpg",
          "answer": "The root scene is broad and quiet."
        }
      ]
    ],
    [
      [
        {
          "image_path": "/your_dataset_root/root/child1/child1_img.jpg",
          "answer": "Child1 shows a green field in the scene."
        }
      ]
    ],
    [
      [
        {
          "image_path": "/your_dataset_root/root/child1/grandchild1/grandchild1_img.jpg",
          "answer": "Grandchild1 describes a tree standing in the green field."
        }
      ]
    ],
    [
      [
        {
          "image_path": "/your_dataset_root/root/child2/child2_img.jpg",
          "answer": "Child2 reveals a river passing by."
        }
      ]
    ],
    [
      [
        {
          "image_path": "/your_dataset_root/root/child2/grandchild2/grandchild2_img.jpg",
          "answer": "Grandchild2 shows a boat floating on the river."
        }
      ]
    ]
  ]
  ```

* **Key requirements:**

  * Each image entry must contain its absolute path (`image_path`) and caption (`answer`).
  * The directory structure in `image_path` must match your on-disk folder hierarchy to correctly infer parent-child relationships.


**Run the Multi-Level Caption Merging Script**

After preparing your JSON file, merge all hierarchical captions with:

```bash
python recursive_merge_caption.py \
    --dataset_root /path/to/your_dataset_root \
    --output_file /path/to/output_merged.json \
    --similarity_threshold 0.2

```

* **`--dataset_root`**: The path to the root folder that contains all root directories (e.g., /your_dataset_root).
* **`--output_file`**: Path to save the output JSON file with the final, recursively merged captions (the result will contain the root image and the merged description).
* **`--similarity_threshold`**: Cosine similarity threshold for caption replacement/merging (default is `0.2`).


2. **Generate VQA Pairs**: Based on the merged caption, five VQA pairs are created. The prompt used to generate these pairs is as follows:
    ```
    Generate five questions and corresponding answers based on the provided caption, covering these aspects: Objects, Relationship, Style, Scene, and Details.
    ```



---




### 3. checkpoint
See huggingface.

### 4. Training (LLaMA-Factory Integration)
For training your own model or fine-tuning OURO with custom datasets, we use the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) repository. This tool allows you to easily create datasets and select sub-regions for training.

- **Steps to Train**:
    1. Install `LLaMA-Factory`:
        ```bash
        git clone https://github.com/hiyouga/LLaMA-Factory.git
        cd LLaMA-Factory
        pip install -r requirements.txt
        ```

    2. Use the provided scripts to create custom datasets from your own image sources.

    3. Choose the sub-regions (generated by the multi-level RPN) that will be used for training.

    4. Train the model using the specified configuration in the `LLaMA-Factory` repository.

## Citation
If you use this code or approach in your research, please cite our paper:

```bibtex
@inproceedings{xu2025ouro,
  title={OURO: A Self-Bootstrapped Framework for Enhancing Multimodal Scene Understanding},
  author={Xu, Tianrun and Chen, Guanyu and Li, Ye and Xi, Yuxin and Mu, Zeyu and Wang, Ruichen and Zhang, Tianren and Gao, Haichuan and Chen, Feng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={18240--18251},
  year={2025}
}
