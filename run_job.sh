#!/bin/bash
#SBATCH --job-name=TFM_mobilenetv4        # Job name for identification
#SBATCH --output=acnn/logs_slurm/output_%j.out # File to store standard output
#SBATCH --error=acnn/logs_slurm/error_%j.err   # File to store error output
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
cd $HOME/TFM/acnn

# 4. Execute the training script
echo "Initiating MobileNetV4 training..."
python main.py --config config.json
echo "Training finished."