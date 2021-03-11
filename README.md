# BM_GPU



python -m ipykernel install --user --name=notebook_analysis
conda activate notebook_analysis

conda env export > environment.yml
conda env create -f environment.yml