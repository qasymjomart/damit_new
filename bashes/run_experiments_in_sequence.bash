#!/bin/bash

# Function to check if any GPU in the specified list has at least 10 GB of free memory
check_gpu_free_threshold() {
  required_free_memory=10240  # 10 GB in MB
  gpus="5 6 7"

  for gpu in $gpus; do
    free_memory=$(nvidia-smi --query-gpu=memory.free --id=$gpu --format=csv,noheader,nounits)
    if [ "$free_memory" -ge "$required_free_memory" ]; then
      echo "GPU $gpu has at least 10 GB of free memory: $free_memory MB."
      export CUDA_VISIBLE_DEVICES=$gpu  # Set CUDA_VISIBLE_DEVICES to the available GPU
      return 0  # Found a suitable GPU
    fi
  done

  echo "No GPUs have at least 10 GB of free memory."
  return 1  # No suitable GPU found
}

# Function to execute a Python command after checking for GPU availability
run_python_command() {
  python_command=$1  # The Python command to execute

  # Wait until a GPU with at least 10 GB free memory is found
  while ! check_gpu_free_threshold; do
    echo "Waiting for a GPU with at least 10 GB of free memory for command: $python_command..."
    sleep 300  # Check every 5 minutes
  done

  # Run the Python command in the background
  echo "Running command: $python_command on GPU $CUDA_VISIBLE_DEVICES..."
  eval "$python_command" &
}

# List of Python commands to execute (split into multiple lines for clarity)
commands=(
  "python kfold_train.py  --config_file configs/config.yaml  --savename vpt_deep_epochs100_p50_pd01_cnad_01  --classes_to_use CN AD  --mode source_only  --source ADNI2  --target ADNI2 --seed 5426 --devices 5 --epochs 50 --optimizer SGD --lr 0.05 --patch_size 16 --batch_size 4 --use_aug  --use_pretrained checkpoints/p_noaug_mae75_BRATS2023_IXI_OASIS3__pretraining_seed_8456_999_077000.pth.tar --use_vpt  --vpt_deep  --num_prompt_tokens 50 --prompt_drop_rate 0.1"
  
  "python kfold_train.py  --config_file configs/config.yaml  --savename vpt_deep_epochs100_p50_pd01_cnad_01  --classes_to_use CN AD  --mode source_only  --source ADNI1  --target ADNI1 --seed 4837 --devices 5 --epochs 50 --optimizer SGD --lr 0.05 --patch_size 16 --batch_size 4 --use_aug  --use_pretrained checkpoints/p_noaug_mae75_BRATS2023_IXI_OASIS3__pretraining_seed_8456_999_077000.pth.tar --use_vpt  --vpt_deep  --num_prompt_tokens 50 --prompt_drop_rate 0.1"
  
  "python kfold_train.py  --config_file configs/config.yaml  --savename vpt_deep_sgd0005_p50_pd01_cnad_01    --classes_to_use CN AD  --mode source_only  --source ADNI2  --target ADNI2 --seed 5426 --devices 6 --epochs 50 --optimizer SGD --lr 0.005 --patch_size 16 --batch_size 4 --use_aug  --use_pretrained checkpoints/p_noaug_mae75_BRATS2023_IXI_OASIS3__pretraining_seed_8456_999_077000.pth.tar --use_vpt  --vpt_deep  --num_prompt_tokens 50 --prompt_drop_rate 0.1"

  "python kfold_train.py  --config_file configs/config.yaml  --savename vpt_deep_sgd0005_p50_pd01_cnad_01 --classes_to_use CN AD  --mode source_only  --source ADNI1  --target ADNI1 --seed 4837 --devices 6 --epochs 50 --optimizer SGD --lr 0.005 --patch_size 16 --batch_size 4 --use_aug  --use_pretrained checkpoints/p_noaug_mae75_BRATS2023_IXI_OASIS3__pretraining_seed_8456_999_077000.pth.tar --use_vpt  --vpt_deep  --num_prompt_tokens 50 --prompt_drop_rate 0.1"
    
  "python kfold_train.py  --config_file configs/config.yaml  --savename vpt_deep_p100_pd01_cnad_01  --classes_to_use CN AD --mode source_only  --source ADNI2  --target ADNI2  --seed 5426 --devices 4 --epochs 50 --optimizer SGD --lr 0.05 --patch_size 16 --batch_size 4 --use_aug  --use_pretrained checkpoints/p_noaug_mae75_BRATS2023_IXI_OASIS3__pretraining_seed_8456_999_077000.pth.tar --use_vpt  --vpt_deep  --num_prompt_tokens 100 --prompt_drop_rate 0.1"
    
  "python kfold_train.py  --config_file configs/config.yaml  --savename vpt_deep_p100_pd01_cnad_01  --classes_to_use CN AD --mode source_only  --source ADNI1  --target ADNI1  --seed 4837 --devices 4 --epochs 50 --optimizer SGD --lr 0.05 --patch_size 16 --batch_size 4 --use_aug  --use_pretrained checkpoints/p_noaug_mae75_BRATS2023_IXI_OASIS3__pretraining_seed_8456_999_077000.pth.tar --use_vpt  --vpt_deep  --num_prompt_tokens 100 --prompt_drop_rate 0.1"
)

# Execute each Python command in parallel
for cmd in "${commands[@]}"; do
  run_python_command "$cmd"
done

# Wait for all background processes to complete
wait

echo "All commands executed successfully."
