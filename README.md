# BM_GPU



python -m ipykernel install --user --name=notebook_analysis
conda activate notebook_analysis

conda env export > environment.yml
conda env create -f environment.yml



%load_ext autoreload
________________________________
%autoreload 0 - disables the auto-reloading. This is the default setting.
%autoreload 1 - it will only auto-reload modules that were imported using the %aimport function (e.g %aimport my_module). It’s a good option if you want to specifically auto-reload only a selected module.
%autoreload 2 - auto-reload all the modules. Great way to make writing and testing your modules much easier.