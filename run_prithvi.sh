#!/bin/bash
#SBATCH --job-name=TFM_prithvi        # Job name for identification
#SBATCH --output=src/logs_slurm/output_%j.out # File to store standard output
#SBATCH --error=src/logs_slurm/error_%j.err   # File to store error output
#SBATCH -N 1                              # Request 1 node
#SBATCH -c 32                             # Request 32 CPU cores (mandatory FT3 ratio)
#SBATCH --mem=120G                        # Request 120 GB of RAM
#SBATCH --time=06:00:00                   # Maximum runtime
#SBATCH --gres=gpu:1                      # Request 1 GPU

# 1. Load Python module
module load python/3.10.8

# 2. Activate virtual environment
source $STORE/env_tfm/bin/activate

# 3. Move to code directory
cd $HOME/TFM/src

# 4. Set environment variable for PyTorch CUDA memory allocation (avoid memory fragmentation)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 5. Execute the training script
echo "Initiating Prithvi training..."
python main.py --config config.json --models prithvi
echo "Training finished."