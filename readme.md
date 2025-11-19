# F21BC 2025 GROUP29
# Train NN using PSO Course Work

# In order to run a sample config and experiments suite
# Run the following unhashed commands

# create virtual environment using a specific python version
py -3.13 -m venv .venv

# OR create virtual environment using default python version
# py -m venv .venv

# activate virtual environment 
.\.venv\Scripts\activate

# install packages required for the solution from requirements.txt file
pip install -r requirements.txt

# run main which runs a sample config (hard-coded in main)
python src/main.py

# Run experiments suite which run a set of configurations (in experiments/config.yaml)
python experiments/main.py

# Run jupyter lab to view experiments result
jupyter lab

# open result_viewer.ipynb