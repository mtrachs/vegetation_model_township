#!/bin/sh
#SBATCH --job-name=test
#SBATCH -c 13
#SBATCH -partition=high
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andria.dawson@gmail.com
#SBATCH --workdir=/accounts/gen/vis/adawson/Documents/projects/vegetation_model_township
#SBATCH --error="slurm-%j.err"
#SBATCH --output="slurm-%j.out"

export OMP_NUM_THREADS=20
srun ./veg_ts_od_mpp_2171.exe sample num_warmup=200 \
        num_samples=1000 \
        save_warmup=1 \
        data file=township_data_13_taxa_6796_cells_120_knots.dump \
        output file=veg_township_output.csv
