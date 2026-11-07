#!/bin/bash
#SBATCH --job-name=TFM_mobilenetv4        # Job name for identification
#SBATCH --output=logs_slurm/output_%j.out # File to store standard output
#SBATCH --error=logs_slurm/error_%j.err   # File to store error output
#SBATCH -N 1                             # Request 1 node
#SBATCH -c 8                             # Request 8 CPU cores (for your 4 num_workers)
#SBATCH --mem=120G                       # Request 120 GB of RAM (essential for loading the image quarters)
#SBATCH --time=12:00:00                   # Maximum runtime (12 hours). Resources are released when the job finishes.
#SBATCH --gres=gpu:1                     # Request 1 GPU

# 1. Load Python module
module load python/3.10.8

# 2. Activate virtual environment
source $STORE/env_tfm/bin/activate

# 3. Move to code directory
cd $HOME/TFM

# 4. (Optional) Create Slurm logs directory if it doesn't exist
mkdir -p logs_slurm

# 5. Execute the training script
echo "Initiating MobileNetV4 training..."
python main.py --config config.json
echo "Training finished."