# ##################### F21BC 2025 GROUP29 
# ##################### Train NN using PSO Course Work

# ########### Build Virtual environment
# Create virtual environment, activate it and install packages
py -3.13 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# ########### Run Sample Config
# Run PSO NN training using a sample configurations hard-coded in main
python src/main.py

# ########### Run Experiments Suite
# Run experiments suite which uses a set of configurations (in experiments/config.yaml)
# Then view results (logged into experiments/result.json) from Jupyter notebook
python execute_exp_suite.py
jupyter lab
# open experiments/result_viewer.ipynb


# ############ View Report Result
# To view result of already executed experiments mentioned in the report 
run jupyter lab
# open experiments/report_notebooks/fixed_budget/result_viewer.ipynb
# open experiments/report_notebooks/vel_coeffs/result_viewer.ipynb
# open experiments/report_notebooks/nn_arch/result_viewer.ipynb