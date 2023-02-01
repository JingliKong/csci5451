#!/bin/bash -l
### Charge to class group time
#SBATCH -A csci5451
### Max runtime
#SBATCH --time=0:00:30
### Number of procs requested
#SBATCH --ntasks=32
### Max memory 
#SBATCH --mem=1g
### Queue to submit to
#SBATCH -p small
### Which events to mail about and which email address to use 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kauffman@umn.edu
cd ~/04-mpi-code
module load ompi
mpirun -np 32 ./a.out > testrun
