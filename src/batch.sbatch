#!/bin/bash
#SBATCH --mail-type=FAIL
#SBATCH --time=99:59:59
module load singularity/3.2.1
singularity exec MWE.simg python3 SFC_mig_simulator.py
