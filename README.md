# FEWTRANS: Benchmarking Few-shot Transferability

This repository contains the official implementation of the paper: **"Benchmarking Few-shot Transferability of Pre-trained Models with Improved Evaluation Protocols"**.

FEWTRANS is a comprehensive benchmark for few-shot transfer learning, featuring 10+ diverse datasets and the **Hyperparameter Ensemble (HPE)** protocol to address the "validation set illusion" in data-scarce scenarios.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Frankluox/FewTrans.git
   cd FewTrans
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) For CLIP-based models:
   ```bash
   pip install ftfy regex tqdm
   pip install git+https://github.com/openai/CLIP.git
   ```

---

## Dataset Preparation

We support multiple datasets including ILSVRC, Omniglot, Aircraft, CUB, etc. 

**Recommendation**: To ensure compatibility with the provided scripts, we recommend creating a symbolic link named `data` in the project root:
```bash
ln -s /path/to/your/datasets ./data
```

### Original Meta-Dataset Setup
Please follow the instructions in the `prepare_datasets.py` for downloading and converting the datasets:

- **ILSVRC 2012**: Download and extract into `ILSVRC2012_img_train/`.
- **Omniglot**: Download `images_background.zip` and `images_evaluation.zip`.
- **Aircraft**: Download `fgvc-aircraft-2013b.tar.gz`.
- **Quick Draw**: Download `.npy` files from Google Cloud.
- **Other Datasets**: CUB, DTD, Fungi, VGG Flower, Traffic Signs, MSCOCO, MNIST, CIFAR10/100.

For conversion scripts, use:
```bash
python -m prepare_datasets.py --data_src_path=<src> --data_dst_path=<dst> --process_<dataset>=1
```
---

## Usage (FEWTRANS Protocol)

The FEWTRANS evaluation follows a two-stage protocol: Hyperparameter Search and Final Evaluation.

### 1. Hyperparameter Search (HPE Protocol)
The **Hyperparameter Ensemble (HPE)** protocol is used to find stable hyperparameter "centers" (learning rates, epochs) before final testing. Run the iterative search script:

```bash
python auto_find.py
```
*Note: You can modify `dataset_idx_list` and `model_name` inside `auto_find.py` to target specific experiments.*

### 2. Final Evaluation
Once the hyperparameters are identified, run the full benchmark using the automated shell script. This script handles YAML generation and execution:

```bash
# Usage: bash group_test.sh [GPU_ID] [MODEL_NAME]
bash group_test.sh 0 my_experiment
```
The `group_test.sh` script automates:
- Generating configs via `write_yaml_test_with_arg_visual_only.py` (for visual-only) or `write_yaml_test_with_arg.py`.
- Executing `main.py` across the benchmark suite (10 datasets).

---

## Core Components

- `auto_find.py`: Implements the iterative hyperparameter search for the HPE protocol.
- `group_test.sh`: Automated evaluation script for the benchmark datasets.
- `models/fewshot_finetune_ensemble.py`: Logic for ensemble-based few-shot fine-tuning.
- `architectures/classifier/prompt_tuning_visualonly_cosine.py`: Visual-only prompt tuning implementation.

---


## Acknowledgements
This codebase is built upon [A Closer Look at Few-shot Classification Again (ICML 2023)](https://github.com/Frankluox/CloserLookAgain).