# AVS-Net

This repository contains an implementation of AVS-Net: Attention-based Variable Splitting Network for P-MRI Acceleration using PyTorch.

## Preliminary Steps

### 1. Download the knee MRI Acceleration dataset (Optional)

For those who require an independent download of the knee dataset, the `git lfs` tool can be utilized. The dataset is available on the `huggingface` datasets platform. Use the following commands:

```bash
# Ensure git-lfs is installed (https://git-lfs.com)
git lfs install
git clone -j8 git@hf.co:datasets/AVS-Net/knee_fast_mri
```

### 2. Clone the source code and dataset submodules

To conveniently clone both the source code and the dataset together, use submodules. The following command combines the dataset download and source code cloning in one step, effectively skipping the step \[1\] described above.

```bash
# Ensure git-lfs is installed (https://git-lfs.com)
git lfs install
git clone --recurse-submodules -j8 https://github.com/AVS-Net/AVS-Net.git
```

### 3. Install dependencies

Ensure Python version 3.8 or later is installed. Dependencies can be installed either using pip or conda:

```bash
pip install pytorch=1.7 torchvision matplotlib h5py scipy scikit-image tensorboard
```

or

```bash
conda install pytorch=1.7 torchvision torchaudio -c pytorch
conda install matplotlib h5py scipy scikit-image tensorboard
```

## Training AVS-Net

To train the AVS-Net on a general-purpose environment, single NVIDIA A100 GPU on an amd64 platform can be used:

```bash
cd avs-net && CUDA_VISIBLE_DEVICES=0 python avs-net.py
```

### Logging training with TensorBoard

To monitor the training process, visit localhost:6006 in your browser. Use the following command to start Tensorboard:

```bash
tensorboard --logdir tensorboard_log
```

## Working on the BlueBEAR HPC

Please note that BlueBEAR is an IBM Power9 Series High Performance Computing system with ppcle-64 architecture.

### General purpose usage on the terminal

After allocating requisite computational resources on a given node using `Slurm`, execute the following command:

```bash
module purge; module load bear-apps/2020b PyTorch &> /dev/null 
cd avs-net && CUDA_VISIBLE_DEVICES=0 python avs-net.py
```

Please be mindful that the connection can be interrupted at any time, which may result in the termination of the training process.

### Utilizing Multiplexer with TMUX

For handling deep learning tasks in the background, especially during debugging, it is recommended to use `tmux` or `screen`.

Here we provide two different approaches to using `tmux`:

```bash
tmux new -s avs-net
module purge; module load bear-apps/2020b PyTorch &> /dev/null 
cd avs-net && CUDA_VISIBLE_DEVICES=0 python avs-net.py
```

or

```bash
tmux -CC # This will open a new terminal
module purge; module load bear-apps/2020b PyTorch &> /dev/null 
cd avs-net && CUDA_VISIBLE_DEVICES=0 python avs-net.py
# You can now close the laptop and return in 2 days
tmux -CC attach
```
