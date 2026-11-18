# S5000 / MUSA training

This opt-in entry uses upstream FastWAM commit
`7faa71108368fbb3b6885649f112af607427a2d4` and the official
`libero_uncond_2cam224_1e-4` training task. It targets the previously working
TrainingSuite 2.1.7 environment (Python 3.10.12, torch/torch_musa 2.7.1.post1,
DeepSpeed 0.19.3+3791c4da, Accelerate 1.12.0).

## Compatibility changes

- Select `musa:LOCAL_RANK` for training and the MUSA DeepSpeed accelerator.
- Compute timestep embeddings in FP32. Generate RoPE frequencies with the
  upstream CPU function, then use FP32 real cos/sin pairs on device instead
  of FP64/complex operations. This is a precision adaptation, not a claim of
  bitwise equality with the upstream FP64 path.
- Run the same batched VAE encoder eagerly, retaining its normalization and
  output clone, because the upstream `cudagraphs` backend is CUDA-specific.

The CUDA entry, trainer, optimizer, masking, gradient clipping, loss,
checkpointing and official task configs are unchanged. The MUSA entry is
scoped to this LIBERO training path; it does not claim validation of compiled
training, evaluation, other tasks or other torch_musa versions. The image
already provides torchvision VideoReader; no replacement decoder is needed.

## Environment and inputs

Use TrainingSuite 2.1.7 with the MUSA container runtime and sufficient shared
memory (the validation uses 32 GiB). Preserve its accelerator packages. Do
not run the upstream default dependency installation in this environment:
`pyproject.toml` pins CUDA packages. Instead, inside the image:

```bash
python -m pip install --no-deps -r requirements-musa.in
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

Follow the upstream README for LIBERO LeRobot data, Wan2.2-TI2V-5B weights,
the ActionDiT initialization and cached text embeddings. Existing prepared
inputs can be mounted read-only; no copy is required. The command below
assumes they are exposed under `/assets` in the layout used by the official
preparation steps. Adjust these path overrides for your own input location.

## Eight-GPU validation command

From the repository root:

```bash
export DS_ACCELERATOR=musa
export DIFFSYNTH_MODEL_BASE_PATH=/assets/checkpoints
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 8 --num_cpu_threads_per_process 4 \
  scripts/train_musa.py \
  task=libero_uncond_2cam224_1e-4 \
  output_dir=./runs/musa-validation \
  max_steps=50 batch_size=16 gradient_accumulation_steps=1 log_every=1 \
  'data.train.dataset_dirs=[/assets/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot,/assets/data/libero_mujoco3.3.2/libero_object_no_noops_lerobot,/assets/data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot,/assets/data/libero_mujoco3.3.2/libero_10_no_noops_lerobot]' \
  data.train.text_embedding_cache_dir=/assets/data/text_embeds_cache/libero \
  model.action_dit_pretrained_path=/assets/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  +data.train.video_backend=pyav
```

This is BF16, ZeRO-1, GBS 128, with 9 actual video frames, two 224x224
cameras concatenated horizontally, and 32 action steps. A step is an optimizer
step. Reserve space for the final model and optimizer checkpoint (the prior
8-GPU run required approximately 100 GB). Use a new output directory; check
GPU occupancy before running. Stop and investigate if OOM occurs.

## Validation (2026-11-18)

The cleaned entry completed 50 optimizer steps on eight S5000 GPUs using the
command above, with driver 3.3.10-server and TrainingSuite 2.1.7 image ID
`67eafc3ad1ce409cb61996b9f86d7e8893c225d0e802abe8589ef4c945219b5c`.
Exit code was 0; all 50 logged losses were finite (final loss 0.5047).
The weight checkpoint and trainer state both report step 50, and all eight
optimizer shard archives passed structural checks. Resume training was not
part of this validation. GPU-process monitoring found no foreign processes.

The interval from completion of step 11 to completion of step 50 was
2.1081 seconds/step, or 60.72 samples/second at GBS 128. This includes the
real data pipeline and logging, and excludes final checkpoint saving; it is
a single validation run, not a controlled cross-device performance result.
CPU reference checks also covered 1D action and 3D video RoPE forward and
backward, and timestep embeddings over positions 0 through 999.
