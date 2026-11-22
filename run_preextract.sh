#!/bin/bash
#SBATCH --job-name=TFM_preextract           # Job name for identification
#SBATCH --output=src/logs_slurm/output_%j.out # Standard output log
#SBATCH --error=src/logs_slurm/error_%j.err   # Error output log
#SBATCH -N 1                                # Request 1 node
#SBATCH -c 8                                # Request 8 CPU cores
#SBATCH --mem=32G                           # Request 32 GB of RAM
#SBATCH --time=02:00:00                     # Maximum runtime

# 1. Load Python module
module load python/3.10.8

# 2. Activate virtual environment
source $STORE/env_tfm/bin/activate

# 3. Move to code directory
cd $HOME/TFM/src

# 4. Run patch pre-extraction script
echo "Initiating patch pre-extraction..."
python preextract_patches.py --config config.json
echo "Patch pre-extraction finished."