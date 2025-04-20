#!/bin/bash --login

#SBATCH --time=3:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --qos=cs
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=80G   # memory per CPU core
#SBATCH -J "Inference on SAT model"   # job name
#SBATCH --mail-user=chaseez@byu.edu   # email address


# Set the max number of threads to use for programs using OpenMP. Should be <= ppn. Does nothing if the program doesn't use OpenMP.
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
nvidia-smi

python main.py -e 'canary' -n 100