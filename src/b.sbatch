#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like so ##SBATCH
#SBATCH --job-name=ofanan_job
#SBATCH --mail-type=ALL
#SBATCH --mail-user=itamar.cohen@polito.it
#SBATCH --partition=global
#SBATCH --time=00:10:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --output=job_name_%j.log
#SBATCH --mem-per-cpu=1024M

# This is a comment. Below the execution commands
whoami
python ./hello_world.py


