#!/bin/bash

#SBATCH --job-name=Aug2_CRNN_Model	# Job name
#SBATCH --output=jobsOut/job.%j.out	# Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=8		# Scheule 8 cores (includes hyperthreading)
#SBATCH --mem=120G
#SBATCH --gres=gpu:a30:1		# Schedule a GPU, or more with gpu:2 etc
#SBATCH --time=50:00:00			# Run time (hh:mm:ss) - run for one hour max
#SBATCH --partition=brown		# Run on Brown queue
#SBATCH --mail-type=FAIL,END		# Send an email when the job is finishes or fails


echo "Running on $(hostname):"
#nvidia-smi

module load singularity

pip install tensorcross
singularity exec --nv /opt/itu/containers/tensorflow/tensorflow-23.05-tf2-py3.sif python ClusterTrainingAug2.py

