# Holosoma Motion Retargeting

This repository provides tools for retargeting human motion data to humanoid robots. It supports multiple data formats (smplh, mocap, lafan) and task types including robot-only motion, object interaction, and climbing.

**Data Requirements**: The retargeting pipeline requires motion data in world joint positions. For custom data, you need to prepare world joint positions in shape `(T, J, 3)` where T is the number of frames and J is the number of joints, and modify `demo_joints` and `joints_mapping` defined in `config_types/data_type.py`.

## Single Sequence Motion Retargeting

```bash
# Robot-only (OMOMO)
python examples/robot_retarget.py --data_path demo_data/OMOMO_new --task-type robot_only --task-name sub3_largebox_003 --data_format smplh --retargeter.debug --retargeter.visualize

# Object interaction (OMOMO)
python examples/robot_retarget.py --data_path demo_data/OMOMO_new --task-type object_interaction --task-name sub3_largebox_003 --data_format smplh --retargeter.debug --retargeter.visualize

# Climbing
python examples/robot_retarget.py --data_path demo_data/climb --task-type climbing --task-name mocap_climb_seq_0 --data_format mocap --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf --retargeter.debug --retargeter.visualize
```

**Note**: Add `--augmentation` to run sequences with augmentation. You must first run the original sequence before adding augmentation.

## Batch Processing for Motion Retargeting

```bash
# Robot-only (OMOMO)
python examples/parallel_robot_retarget.py --data-dir demo_data/OMOMO_new --task-type robot_only --data_format smplh --save_dir demo_results_parallel/g1/robot_only/omomo --task-config.object-name ground

# Object interaction (OMOMO)
python examples/parallel_robot_retarget.py --data-dir demo_data/OMOMO_new --task-type object_interaction --data_format smplh --save_dir demo_results_parallel/g1/object_interaction/omomo --task-config.object-name largebox

# Climbing
python examples/parallel_robot_retarget.py --data-dir demo_data/climb --task-type climbing --data_format mocap --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf --task-config.object-name multi_boxes --save_dir demo_results_parallel/g1/climbing/mocap_climb
```

**Note**: Add `--augmentation` to run original sequences and sequences with augmentation (for object interaction and climbing tasks).

## Data Preparation

We provide `demo_data/` for fast testing. To test on more motion sequences, please follow the instructions below to download and prepare the data.

### OMOMO

Our pipeline uses the processed dataset by InterMimic. The data format differs from the original OMOMO dataset.

1. Download the processed OMOMO data from [this link](https://drive.google.com/file/d/141YoPOd2DlJ4jhU2cpZO5VU5GzV_lm5j/view)
2. Extract the downloaded folder to `demo_data/OMOMO_new`

The data should contain `.pt` files.

### LAFAN

#### Download the Original LAFAN Data

1. Download [lafan1.zip](https://github.com/ubisoft/ubisoft-laforge-animation-dataset/blob/master/lafan1/lafan1.zip) by clicking "View Raw"
2. Put `lafan1.zip` in your designated data folder and uncompress it to `DATA_FOLDER_PATH/lafan`
3. The file structure should be `demo_data/lafan/*.bvh`

#### Convert the Original LAFAN Data Format for Motion Retargeting

We need some data processing files from the [LAFAN GitHub repo](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).

```bash
cd holosoma_retargeting/data_utils/
git clone https://github.com/ubisoft/ubisoft-laforge-animation-dataset.git
mv ubisoft-laforge-animation-dataset/lafan1 .
python extract_global_positions.py --input_dir DATA_FOLDER_PATH/lafan --output_dir ../demo_data/lafan
```

This will convert the BVH files to `.npy` format with global joint positions.

**Note**: For LAFAN data, you need to relax the foot sticking constraint by setting `--retargeter.foot-sticking-tolerance` (default is stricter). You can adjust this tolerance number based on your data quality and retargeting results.

#### Single Sequence Retargeting on LAFAN

```bash
python examples/robot_retarget.py --data_path demo_data/lafan --task-type robot_only --task-name dance2_subject1 --data_format lafan --task-config.ground-range -10 10 --save_dir demo_results/g1/robot_only/lafan --retargeter.debug --retargeter.visualize --retargeter.foot-sticking-tolerance 0.02
```

#### Batch Processing for Motion Retargeting on LAFAN

```bash
python examples/parallel_robot_retarget.py --data-dir demo_data/lafan --task-type robot_only --data_format lafan --save_dir demo_results_parallel/g1/robot_only/lafan --task-config.object-name ground --task-config.ground-range -10 10 --retargeter.foot-sticking-tolerance 0.02
```

### AMASS SMPL-X

#### Download the Original AMASS Data

1. Follow the [AMASS](https://amass.is.tue.mpg.de/) instructions to download the original AMASS data
2. The AMASS data structure should be `/path/to/amass/dataset_name/subject_name/*.npz`

#### Download SMPL-X Models

1. Follow the [SMPL-X](https://smpl-x.is.tue.mpg.de/index.html) instructions to download SMPL-X models
2. For AMASS data, we tested on SMPL-X N (neutral) format
3. The SMPL-X models structure should be `/path/to/models/smplx/SMPLX_NEUTRAL.npz`

#### Convert the Original AMASS SMPL-X Data Format for Motion Retargeting

We provide `data_utils/prep_amass_smplx_for_rt.py` for converting AMASS SMPLX data to the format required for motion retargeting.

```bash
# Install dependencies
cd holosoma_retargeting/data_utils/
git clone https://github.com/nghorbani/human_body_prior.git
pip install tqdm dotmap PyYAML omegaconf loguru
cd human_body_prior/
python setup.py develop
cd ../

# Run data processing
python prep_amass_smplx_for_rt.py \
  --amass-root-folder /path/to/amass \
  --output-folder /path/to/output \
  --model-root-folder /path/to/models
```

This will convert the AMASS `.npz` files to `.npz` format with global joint positions and height information.

**Note**: You can optionally specify `--subdataset-folder` to process only a specific subdataset (e.g., `HumanEva`). If not specified, it will process all datasets recursively.

#### Single Sequence Retargeting on AMASS SMPL-X

```bash
python examples/robot_retarget.py --data_path demo_data/amass_smplx_processed --task-type robot_only --task-name HumanEva_S3_Jog_1_stageii --data_format smplx --task-config.ground-range -10 10 --save_dir demo_results/g1/robot_only/amass_smplx --retargeter.debug --retargeter.visualize
```

#### Batch Processing for Motion Retargeting on AMASS SMPL-X

```bash
python examples/parallel_robot_retarget.py --data-dir demo_data/amass_smplx_processed --task-type robot_only --data_format smplx --save_dir demo_results_parallel/g1/robot_only/amass_smplx --task-config.object-name ground --task-config.ground-range -10 10
```

## Check Visualizations of Saved Retargeting Results

```bash
# Visualize object-interaction results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --object_urdf models/largebox/largebox.urdf \
    --qpos_npz demo_results_parallel/g1/object_interaction/omomo/sub3_largebox_003_original.npz

# Visualize climbing results
python viser_player.py --robot_urdf models/g1/g1_29dof_spherehand.urdf \
    --object_urdf demo_data/climb/mocap_climb_seq_0/multi_boxes.urdf \
    --qpos_npz demo_results_parallel/g1/climbing/mocap_climb/mocap_climb_seq_0_original.npz

python viser_player.py --robot_urdf models/g1/g1_29dof_spherehand.urdf \
    --object_urdf demo_data/climb/mocap_climb_seq_0/multi_boxes_scaled_0.74_0.74_0.89.urdf \
    --qpos_npz demo_results_parallel/g1/climbing/mocap_climb/mocap_climb_seq_0_z_scale_1.2.npz

# Visualize robot only results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results_parallel/g1/robot_only/omomo/sub3_largebox_003_original.npz

# Visualize LAFAN robot only results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results/g1/robot_only/lafan/dance2_subject1.npz

# Visualize AMASS results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results/g1/robot_only/amass_smplx/HumanEva_S3_Jog_1_stageii.npz

# Visualize AMASS results
python viser_player.py --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz demo_results_parallel/g1/robot_only/amass_smplx/HumanEva_S1_Box_1_stageii_original.npz
```

## Quantitative Evaluation

```bash
# Evaluate robot-object interaction
python evaluation/eval_retargeting.py --res_dir demo_results_parallel/g1/object_interaction/omomo --data_dir demo_data/OMOMO_new --data_type "robot_object"

# Evaluate climbing sequence
python evaluation/eval_retargeting.py --res_dir demo_results_parallel/g1/climbing/mocap_climb --data_dir demo_data/climb --data_type "robot_terrain" --robot-config.robot-urdf-file models/g1/g1_29dof_spherehand.urdf

# Evaluate robot only (OMOMO)
python evaluation/eval_retargeting.py --res_dir demo_results_parallel/g1/robot_only/omomo --data_dir demo_data/OMOMO_new --data_type "robot_only"
```

## Prepare Data for Training RL Whole-Body Tracking Policy

To prepare data for training RL whole-body tracking policies, you need to follow a two-step process:

1. **First, run retargeting** to obtain `.npz` files containing the retargeted robot motion. Use the retargeting commands shown in the sections above (Single Sequence Motion Retargeting or Batch Processing for Motion Retargeting).

2. **Then, run the data conversion code** below to convert the retargeted `.npz` files into the format required for RL training. The conversion script takes the retargeted `.npz` files as input and outputs converted files with the specified frame rate and format.

**Note**: If you run this code on Mac, please use `mjpython` instead of `python`.

### Mac (using mjpython)

```bash
mjpython data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz --output_fps 50 --output_name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz --data_format smplh --object_name "ground" --once

mjpython data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz --output_fps 50 --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz --data_format smplh --object_name "largebox" --has_dynamic_object --once
```

### Robot-Only Setting

```bash
python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/omomo/sub3_largebox_003.npz --output_fps 50 --output_name converted_res/robot_only/sub3_largebox_003_mj_fps50.npz --data_format smplh --object_name "ground" --once

python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/robot_only/lafan/dance2_subject1.npz --output_fps 50 --output_name converted_res/robot_only/dance2_subject1_mj_fps50.npz --data_format lafan --object_name "ground" --once
```

### Robot-Object Setting

```bash
python data_conversion/convert_data_format_mj.py --input_file ./demo_results/g1/object_interaction/omomo/sub3_largebox_003_original.npz --output_fps 50 --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj.npz --data_format smplh --object_name "largebox" --has_dynamic_object --once
```

### OmniRetarget Data

For OmniRetarget data downloaded from HuggingFace, please add `--use_omniretarget_data` for data conversion.

```bash
python data_conversion/convert_data_format_mj.py --input_file OmniRetarget/robot-object/sub3_largebox_003_original.npz --output_fps 50 --output_name converted_res/object_interaction/sub3_largebox_003_mj_w_obj_omnirt.npz --data_format smplh --object_name "largebox" --has_dynamic_object --use_omniretarget_data --once
```

## WorldPoseDataset (FIFA Soccer Broadcast Motion)

This pipeline converts the [WorldPoseDataset](https://github.com/jyuntins/world-pose) FIFA soccer broadcast poses (SMPL format) into training clips for the MESSI whole-body tracking policy on Unitree G1.

### Prerequisites

**WorldPoseDataset**: Download from the [WorldPoseDataset repo](https://github.com/jyuntins/world-pose). Pose `.npz` files should be at `/path/to/FIFA/poses/*.npz`. Each file contains up to 22 players tracked over ~1000 frames.

**SMPL model**: Register at [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/) and download `SMPL_MALE.pkl`. Place it in a directory, e.g. `/path/to/smpl_models/SMPL_MALE.pkl`.

### One-Command Pipeline

Run all 4 steps end-to-end from the MESSI repo root:

```bash
WORLDPOSE_DATA=/path/to/FIFA/poses \
SMPL_MODEL=/path/to/smpl_models/SMPL_MALE.pkl \
OUTPUT_DIR=/path/to/output \
bash demo_scripts/demo_worldpose_pipeline.sh
```

Override any default via env var. The script sources the retargeting conda env automatically (`hsretargeting` by default; set `CONDA_ENV_NAME=<name>` to override).

### Pipeline Steps

The pipeline is orchestrated by `examples/run_worldpose_pipeline.py` and runs 4 steps:

| Step | Script | Output |
|------|--------|--------|
| 1 | `data_utils/prep_worldpose_for_rt.py` | Split players by NaN gaps, run SMPL FK → `(T, 22, 3)` joint clips |
| 2 | `examples/parallel_robot_retarget.py` | Retarget human joints to G1 29-DOF joint space |
| 3 | `data_conversion/convert_data_format_mj.py` | Convert to MuJoCo qpos/qvel format |
| 4 | `data_utils/preprocess_retargeted.py` | Resample to 50 Hz + smooth-start blend |

Run from `src/holosoma_retargeting/holosoma_retargeting/`:

```bash
# Full pipeline
python examples/run_worldpose_pipeline.py \
    --data_dir /path/to/FIFA/poses \
    --smpl_model_path /path/to/smpl_models/SMPL_MALE.pkl \
    --output_dir /path/to/output \
    --smooth_start

# Resume from step 2 (skip step 1 if clips already prepared)
python examples/run_worldpose_pipeline.py \
    --data_dir /path/to/FIFA/poses \
    --smpl_model_path /path/to/smpl_models/SMPL_MALE.pkl \
    --output_dir /path/to/output \
    --start_step 2
```

### Key Parameters

- `--task-config.ground-range -60 60` — soccer pitch is ~105 × 68 m; the default ±60 m covers it
- `--retargeter.foot-sticking-tolerance 0.02` — relaxed foot sticking for dynamic soccer motion (default is stricter)
- `--smooth_start` — prepend a 0.5 s default-pose hold + 0.3 s blend-in to each clip for stable training initialization

### Training the WBT Policy

After the pipeline completes, training-ready clips are at `OUTPUT_DIR/step4_preprocessed/`. Train the whole-body tracking policy:

```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-fast-sac \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_file=/path/to/output/step4_preprocessed
```

### Visualization

All visualization commands run from `src/holosoma_retargeting/holosoma_retargeting/`.

**Step 1 — SMPL skeleton clips** (sanity-check the joint positions before retargeting):

```bash
# Single clip → renders front + side view to mp4
python data_utils/visualize_clips.py \
    --input /path/to/output/step1_prepared/ARG_CRO_220001_p00_s00.npz \
    --output /path/to/videos/

# Whole directory (first 10 clips)
python data_utils/visualize_clips.py \
    --input_dir /path/to/output/step1_prepared/ \
    --output_dir /path/to/videos/ \
    --max_clips 10
```

**Step 2 — Retargeted robot motion** (interactive 3-D viewer via Viser; open browser at the printed URL):

```bash
python viser_player.py \
    --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz /path/to/output/step2_retargeted/<clip_name>_original.npz
```

**Step 4 — Final training clips** (MuJoCo-format, 50 Hz):

```bash
python viser_player.py \
    --robot_urdf models/g1/g1_29dof.urdf \
    --qpos_npz /path/to/output/step4_preprocessed/<clip_name>_original.npz
```

## Custom Human Motion Data Format
Please see the instructions for custom human motion data formats: [ADD_MOTION_FORMAT_README.md](ADD_MOTION_FORMAT_README.md)

## Custom Robot Type
Please see the instructions for retargeting custom robot types: [ADD_ROBOT_TYPE_README.md](ADD_ROBOT_TYPE_README.md)
