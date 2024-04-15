#!/bin/bash
#SBATCH -t 00-06:00:00
#SBATCH -n 1
#SBATCH --gpus=a100_1g.5gb
#SBATCH -p gpu
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH -A brehm-prj-paid

source /home/jamcq/Modules/modulefiles/champsGPU/1.0
#compute-sanitizer --tool memcheck --leak-check full ./vdf.x input.sdf
#compute-sanitizer --tool racecheck ./vdf.x input.sdf
#compute-sanitizer --tool initcheck ./vdf.x input.sdf
#compute-sanitizer --tool synccheck ./vdf.x input.sdf
./vdf.x input.sdf
