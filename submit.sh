#!/bin/sh
#SBATCH --job-name={short}
#SBATCH -c {threads}
#SBATCH --mail-type=ALL
#SBATCH --mail-user=andria.dawson@gmail.com
#SBATCH --workdir=/accounts/gen/vis/adawson/Documents/projects/stepps-prediction/runs/{model}/{run}
#SBATCH --error="slurm-%j.err"
#SBATCH --output="slurm-%j.out"

cp $HOME/Documents/projects/vegetation-model-township/veg_ts_od_mpp_262.cpp .
cp $HOME/Documents/projects/vegetation-model-township/veg_ts_od_mpp_262.exe .

export OMP_NUM_THREADS=20
srun ./{exe} sample num_warmup=200 \
        num_samples=1000 \
        save_warmup=1 \
        data file=township_data_13_taxa_6796_cells_120_knots.dump \
        output file=output.csv
