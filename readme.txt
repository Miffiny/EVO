TSP Project

This project generates point datasets from images, runs Max-Min Ant System
experiments and builds plots from the resulting CSV files.

Layout
----------------------------------------------------------------------------

- src/ contains the C++ core, Python scripts, Makefile and requirements.txt
- data/img/ contains source images and generated preview images
- data/points/ contains generated point CSV files
- data/precomputed/ contains binary datasets consumed by the C++ core
- results/run_results/ contains experiment CSV output
- results/plots_faceted/ contains generated plots

System dependencies:

- Python 3.10 an higher
- A C++17 compiler with OpenMP support (was tested with g++)
- make if you want to build with 'src/Makefile'

Python packages:

- numpy
- Pillow
- pandas
- matplotlib
- seaborn
- scipy
- statsmodels

Install Python dependencies from the project root. You can use a venv
python -m venv .venv
source .venv/bin/activate
python -m pip install -r src/requirements.txt

Or just directly

.venv/bin/python -m pip install -r src/requirements.txt

Build
---------------------------------------------------

Run build commands from src/.

With make

cd src
make

or manually

cd src
g++ -std=c++17 -Wall -Wextra -O3 -fopenmp -o mmas_core main.cpp aco.cpp

Run Order
-----------------------------------------------------

1. Put source images into data/img/.
2. Generate grayscale images, point CSV files, point preview images, and binary
   precomputed datasets:

cd src
python generator.py

3. Build the C++ core if not done already
make

4. Run experiments:

./mmas_core single
./mmas_core ofat
./mmas_core pair
./mmas_core opt

or with a single command
./mmas_core all

5. Draw single-run routes over point preview images:

python path_builder.py

6. Generate plots:

python stats.py ofat
python stats.py pair
python stats.py opt

or with a single command
python stats.py all

C++ Core Usage
-----------------------------------------

from src/:
./mmas_core [single|ofat|pair|opt|all] [pair_param_1 pair_param_2]

Modes:

- single: runs one baseline MMAS execution per dataset and writes the best route
  to results/single_run_results/single_route_<dataset>.csv.
- ofat runs one-factor-at-a-time MMAS experiments and writes
  results/run_results/experiment_results_<dataset>.csv.
- pair: runs pair experiments and writes
  results/run_results/pairs_<param1>_<param2>_<dataset>.csv.
- opt: runs local-optimization post-processing experiments and writes
  results/run_results/opt_results_<dataset>.csv.
- all: runs single, ofat, pair and opt in sequence.

The default is ofat.

Pair mode default is alpa beta:
./mmas_core pair

Pair mode can compare any two supported parameters:

./mmas_core pair alpha beta
./mmas_core pair evaporation tau_gap
./mmas_core pair coverage ants

Supported pair parameters:

- alpha
- beta
- evaporation
- tau_gap
- coverage
- ants
- iterations

Experiment Details
----------------------------------------------

OFAT experiments vary:

- alpha
- beta
- evaporation
- ants
- tau_gap
- coverage
- pheromone update strategy: iteration_best and best_so_far

Pair experiments use the selected two parameters and keep the other baseline
settings fixed.

Optimization experiments run the baseline search once per trial, store the best
routes in a buffer and apply each local optimization as post-processing.

Optimization variants:

- baseline without local optimization
- 2-opt
- or-opt
- node_exchange
- node_exchange + 2-opt

Single Route Rendering
-------------------------------------------

After running ./mmas_core single, render saved routes over generated point
preview images:

python path_builder.py

By default, the script reads:
results/single_run_results/single_route_*.csv

It writes:
results/single_run_results/path_<dataset>.png

Render one route file:
python path_builder.py ../results/single_run_results/single_route_gray_bnw.csv

Use custom paths:
python path_builder.py --images-dir ../data/img --output-dir ../results/single_run_results

Statistics Usage
-------------------------------------------

From src/:
python stats.py [ofat|pair|opt|all] [pair_param_1 pair_param_2]
Modes:

- ofat: reads results/run_results/experiment_results_*.csv and writes plots
  to results/plots_faceted/ofat_experiments/.
- pair: reads
  results/run_results/pairs_<param1>_<param2>_*.csv and writes heatmaps to
  results/plots_faceted/pairs/.
- opt: reads `results/run_results/opt_results_*.csv and writes plots to
  results/plots_faceted/ofat_optimizations/.
- all: generates all supported plot groups.

Stats pair mode uses the same pair parameters as the C++ core:
python stats.py pair alpha beta
python stats.py pair evaporation tau_gap

Optional stats paths:
python stats.py ofat --results-dir ../results/run_results --plots-dir ../results/plots_faceted

