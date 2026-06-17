## RSEE

This is the source code for our paper: **Joint Adaptive Resolution Selection and Conditional Early Exiting for Efficient Video Recognition on Edge Devices**. A brief introduction of this work is as follows:

Given the explosive growth in video content generation, there is a rising demand for efficient and scalable video recognition. Deep learning has shown its remarkable performance in video analytics, by applying 2D or 3D Convolutional Neural Networks (CNNs) across multiple video frames. However, high data quantities, intensive computational costs, and various performance requirements restrict the deployment and application of these video-oriented models on resource-constrained edge devices, e.g., Internet-of-Things (IoT) and mobile devices. To tackle this issue, we propose a joint optimization system RSEE by adaptive Resolution Selection (RS) and conditional Early Exiting (EE) to facilitate efficient video recognition based on 2D CNN backbones. Given a video frame, RSEE firstly determines what input resolution is to be used for processing by the dynamic resolution selector, then sends the resolution-adjusted frame into the backbone network to extract features, and finally determines whether to stop further processing based on the accumulated features of current video at the early-exiting gate. Extensive experiments conducted on benchmark datasets indicate that RSEE remarkably outperforms current state-of-the-art solutions in terms of computational cost (by up to 84.72% on UCF101 and 78.93% on HMDB51) and inference speed (up to 3.18× on UCF101 and 3.50× on HMDB51), while still preserving competitive recognition accuracy (up to 7.81% on UCF101 7.21% on HMDB51). Furthermore, the superiority of RSEE on resource-constrained edge devices is validated on the NVIDIA Jetson Nano, with processing speeds controlled by hyperparameters ranging from about 12 to 60 Frame-Per-Second (FPS) that well enable real-time analysis.

鉴于视频内容生成的爆炸性增长，对高效可扩展视频识别的需求日益迫切。深度学习通过在多帧视频上应用二维或三维卷积神经网络（CNN），在视频分析领域展现出卓越性能。然而高数据量、密集型计算成本及多样化性能要求，限制了这些视频模型在资源受限边缘设备（如物联网和移动设备）的部署应用。为此，我们提出联合优化系统 RSEE，通过自适应分辨率选择（RS）与条件化早退机制（EE），基于二维 CNN 主干网络实现高效视频识别。该系统动态处理视频帧时：首先通过分辨率选择器确定输入分辨率，随后将调整后的帧送入主干网络提取特征，最终根据当前视频累积特征在早退门控节点决定是否终止计算。在基准数据集上的实验表明，RSEE 在计算成本（UCF101 数据集最高降低 84.72%，HMDB51 数据集降低 78.93%）和推理速度（UCF101 提升 3.18 倍，HMDB51 提升 3.50 倍）方面显著优于现有方案，同时保持竞争优势的识别准确率（UCF101 最高达 7.81%，HMDB51 达 7.21%）。此外，在 NVIDIA Jetson Nano 边缘设备上的验证证实，RSEE 可通过超参数调控实现约 12-60 帧/秒的处理速度，充分满足实时分析需求。

This work was published by Big Data Mining and Analytics. Click [here](https://www.sciopen.com/article/10.26599/BDMA.2024.9020093) for our paper.

## Required software

- PyTorch (>= 1.7)
- Torchvision
- NumPy
- ptflops
- tensorboardX
- torchsnooper

## Project Structure

```
RSEE/
├── main_base.py            # Main training and evaluation script
├── test.py                 # Standalone test script
├── data_process.py         # Raw video preprocessing to image frames
├── opts.py                 # Command-line argument definitions
├── start.sh                # Entry-point script to launch training/testing
├── train_ucf101.sh         # Training scripts for UCF101
├── train_hmdb.sh           # Training scripts for HMDB51
├── test.sh                 # Inference script for HMDB51
├── ops/
│   ├── rsee.py             # RSEE model: dynamic resolution selector + early-exit gate
│   ├── parallel_resnet.py  # Multi-resolution parallel ResNet backbone
│   ├── parallel.py         # ModuleParallel wrapper for shared/multi-branch layers
│   ├── basic_ops.py        # Consensus module (avg/max) and basic ops
│   ├── blstm.py            # BLSTM / STAN-LSTM temporal modeling heads
│   ├── dataset.py          # Video frame dataset loader
│   ├── dataset_config.py   # Dataset path/name configuration
│   ├── transforms.py       # Data augmentations & multi-scale pre-processing
│   ├── sal_rank_loss.py    # Saliency-based ranking loss for joint training
│   ├── net_flops_table.py  # FLOPs / feature-dim table for supported backbones
│   ├── my_logger.py        # Dual file + stdout logger
│   └── utils.py            # AverageMeter, accuracy, Recorder, mAP utilities
└── README.md
```

## Core Modules

### RSEE Model (`ops/rsee.py`)

The RSEE model jointly performs adaptive Resolution Selection (RS) and conditional Early Exiting (EE) for efficient video recognition.

**Key attributes:**

| Attribute | Description |
|---|---|
| `num_segments` | Number of sampled frames per video (e.g., 10 or 16) |
| `base_model` | Backbone network name (e.g., `parallel_resnet50`, `resnet50`, `vgg16`) |
| `reso_list` | Candidate input resolutions, e.g., `[224, 168, 112, 84]` |
| `skip_list` | Candidate exit positions (number of processed frames) |
| `policy_backbone` | Lightweight policy network, e.g., `mobilenet_v2` |
| `rescale_to` | Target resolution for backbone pre-training alignment |
| `blstm` / `stan_lstm` | Temporal modeling heads for sequence aggregation |
| `consensus_type` | Frame-level consensus: `avg` or `max` |

**State space** (input to the policy network):

| Index | Feature | Dimension |
|---|---|---|
| 0 | Already-processed frame count | 1 |
| 1 | Predicted entropy / confidence of current frame | 1 |
| 2 | Average entropy of the video so far | 1 |
| 3+ | Visual features of the current frame (offset by `policy_input_offset`) | backbone-dependent |

**Action space:**

| Index | Meaning | Range |
|---|---|---|
| 0 | Selected input resolution | index over `reso_list` |
| 1 | Skip / early-exit decision | `{0, 1}` (continue vs. exit) |

**Key methods:**

- `forward(input)` — Sample a frame, run the policy network, choose a resolution and an exit action, then route the frame through the corresponding backbone branch and return the logits.

- `_get_resolution_dimension()` — Resolve the policy input dimension from the candidate resolution list and the policy backbone.

- `_get_action_dimension()` — Combine the resolution-selection and early-exiting action dimensions.

- `_prepare_policy_net()` — Build the lightweight policy backbone (default: MobileNet-V2) used to extract features for the decision head.

- `_extends_to_multi_models()` — Construct the parallel multi-resolution backbone branches and tie their weights where appropriate.

### ParallelResNet Backbone (`ops/parallel_resnet.py`)

A modified ResNet that exposes intermediate feature maps at multiple spatial resolutions from a single shared backbone, enabling dynamic resolution selection without re-instantiating the network.

**Key methods:**

- `forward(x)` — Forward through the stem and stages, returning a list of feature maps at different scales so the policy module can pick the most appropriate one.

### Temporal Modeling (`ops/blstm.py`)

Provides several sequence heads used on top of the per-frame features:

| Module | Description |
|---|---|
| `BLSTM_IRTENet` | Bi-directional LSTM that aggregates per-frame features into a video-level representation |
| `MaxPooling` | Frame-wise max pooling consensus |
| `AvePooling` | Frame-wise average pooling consensus |
| `STAN_LSTM` | Spatial-temporal attention LSTM for fine-grained temporal modeling |

### Saliency-based Ranking Loss (`ops/sal_rank_loss.py`)

Implements the joint training objective that balances classification accuracy, computational cost (FLOPs), and prediction confidence, enabling end-to-end optimization of the policy network.

### Data Pipeline (`ops/dataset.py`, `ops/transforms.py`)

- `TSNDataSet` — Loads pre-extracted RGB frames of a video, samples `num_segments` frames using the standard TSN sparse sampling strategy, and applies the data transforms.

- `data_process.py` — Converts raw videos into per-frame JPEG images at a uniform sampling rate, which is the expected input format for training and inference.

### Utility Modules

- `net_flops_table.py` — Lookup table of feature dimensions and FLOPs for supported backbones (ResNet, MobileNet-V2, VGG, EfficientNet variants).
- `utils.py` — Training/evaluation helpers: `AverageMeter`, `accuracy`, `Recorder`, `cal_map`, plus checkpoint loading utilities.
- `my_logger.py` — Logger that simultaneously writes to stdout and to a log file.
- `basic_ops.py` — Consensus module and miscellaneous small layers.

## Usage

```
# 1) Prepare data: extract frames from raw videos
python data_process.py

# 2) Train on HMDB51
bash train_hmdb.sh

# 3) Train on UCF101
bash train_ucf101.sh

# 4) Test / inference
bash test.sh <DATA_DIR> <MODEL_DIR>
# Example:
bash test.sh ../HMDB51 logs_RSEE/.../ckpt.best.pth.tar
```

**Common training options (see `opts.py` for the full list):**

| Argument | Description |
|---|---|
| `--arch` | Backbone architecture, e.g., `parallel_resnet50`, `resnet50`, `vgg16` |
| `--num_segments` | Number of frames sampled per video |
| `--reso_list` | Candidate input resolutions, e.g., `224 168 112 84` |
| `--policy_backbone` | Backbone of the policy network, e.g., `mobilenet_v2` |
| `--ada_reso_skip` | Enable joint adaptive resolution & early-exit training |
| `--blstm` | Add a Bi-LSTM temporal head on top of the backbone |
| `--irte_joint` / `--irte_final` | Two-stage training: joint, then re-finetune the final head |
| `--exp_decay` / `--init_tau` | Gumbel-softmax temperature schedule for the policy |
| `--use_gflops_loss` | Add FLOPs as part of the training loss |
| `--accuracy_weight` / `--efficency_weight` | Trade-off between accuracy and efficiency |
| `--data_dir` | Path to the processed dataset (e.g., `../UCF101`) |
| `--log_dir` | Directory to save logs and checkpoints |
| `--test_from` | Path to a checkpoint for evaluation |

## Citation

If you find RSEE useful or relevant to your project and research, please kindly cite our paper:

```
@article{Wang2025,
  author  = {Qingli Wang and Chengwu Yu and Shan Chen and Weiwei Fang and Naixue Xiong},
  title   = {Joint Adaptive Resolution Selection and Conditional Early Exiting for Efficient Video Recognition on Edge Devices},
  year    = {2025},
  journal = {Big Data Mining and Analytics},
  keywords = {deep learning, video analytics, edge intelligence, resolution selection, early exit},
  url     = {https://www.sciopen.com/article/10.26599/BDMA.2024.9020093},
  doi     = {10.26599/BDMA.2024.9020093}
}
```

## Contact

Qingli Wang ([20120418@bjtu.edu.cn](mailto:20120418@bjtu.edu.cn))

> Please note that the open source code in this repository was mainly completed by the graduate student author during his master's degree study. Since the author did not continue to engage in scientific research work after graduation, it is difficult to continue to maintain and update these codes. We sincerely apologize that these codes are for reference only.
