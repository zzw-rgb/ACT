# ACT: Action Chunking with Transformers

This repository contains the ACT policy implementation and two simulated bimanual robot environments:
Transfer Cube and Bimanual Insertion. You can collect scripted demonstrations, train policies, evaluate checkpoints, and visualize collected episodes.

For real-robot usage, install the ALOHA codebase separately and keep its task configuration available to the training script.

## Repo Structure

- `imitate_episodes.py`: train and evaluate ACT or CNNMLP policies.
- `record_sim_episodes.py`: collect scripted demonstration episodes in simulation.
- `constants.py`: task configs, data paths, robot constants, and asset paths.
- `sim_env.py`: MuJoCo + DM Control environments with joint-space control.
- `ee_sim_env.py`: MuJoCo + DM Control environments with end-effector control.
- `scripted_policy`: scripted demonstration policies for simulated tasks.
- `model/`: policy wrapper, model builder, ACT VAE, CNNMLP baseline, backbone, transformer, and position encoding modules.
- `utils/`: data loading, pose sampling, helper functions, plotting, videos, and dataset visualization CLI.
- `assets/`: MuJoCo XML and mesh assets.
- `summarize/`: project reading notes and diagrams.

## Installation

```bash
conda create -n aloha python=3.8.10
conda activate aloha
pip install torchvision
pip install torch
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
pip install mujoco==2.3.7
pip install dm_control==1.0.14
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install ipython
```

You can also create the environment from the included file:

```bash
conda env create -f conda_env.yaml
conda activate aloha
```

Run commands from the repository root.

## Simulated Experiments

Set `DATA_DIR` in `constants.py` before training so each task points to the correct dataset directory.

Collect 50 scripted demonstrations for Transfer Cube:

```bash
python record_sim_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --dataset_dir <data save dir> \
  --num_episodes 50
```

Use `--onscreen_render` if you want to see the simulator while collecting data.

Visualize a collected episode:

```bash
python -m utils.visualization --dataset_dir <data save dir> --episode_idx 0
```

This writes `episode_0_video.mp4` and `episode_0_qpos.png` into the dataset directory.

Train ACT:

```bash
python imitate_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir <ckpt dir> \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --batch_size 8 \
  --dim_feedforward 3200 \
  --num_epochs 2000 \
  --lr 1e-5 \
  --seed 0
```

Evaluate the best validation checkpoint by running the same command with `--eval`:

```bash
python imitate_episodes.py \
  --eval \
  --task_name sim_transfer_cube_scripted \
  --ckpt_dir <ckpt dir> \
  --policy_class ACT \
  --kl_weight 10 \
  --chunk_size 100 \
  --hidden_dim 512 \
  --batch_size 8 \
  --dim_feedforward 3200 \
  --num_epochs 2000 \
  --lr 1e-5 \
  --seed 0
```

Add `--temporal_agg` during evaluation to enable temporal action aggregation. Evaluation videos and `result_policy_best.txt` are saved in the checkpoint directory.

## Useful Import Paths

```python
from model.act import ACTPolicy, CNNMLPPolicy
from utils.data import load_data
from utils.pose import sample_box_pose, sample_insertion_pose
from utils.helpers import compute_dict_mean, set_seed, detach_dict
from utils.visualization import save_videos, plot_history
```

## Notes

- Training and evaluation use `.cuda()`, so a CUDA-capable GPU is expected.
- Checkpoints store model parameters only; evaluation also needs the matching `dataset_stats.pkl` in the same checkpoint directory.
- If you change camera names or episode lengths in `constants.py`, collect compatible data before training.
