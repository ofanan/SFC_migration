#!/bin/bash
#SBATCH --mail-type=FAIL
module load singularity/3.2.1
singularity exec MWE.simg python3 SFC_mig_simulator.py 1
singularity exec MWE.simg python3 SFC_mig_simulator.py 2
