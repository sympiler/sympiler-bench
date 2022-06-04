# Sparse Fusion Benchmark
This directory contains scripts for reproducing results in 
the sparse fusion paper at SC22.  


# How to run the artifact
* The sympiler-bench repository should be cloned: 
```
	git clone --recursive https://github.com/sympiler/sympiler-bench.git
	cd sympiler-bench
	git checkout sparse-fusion
	cd sparse-fusion
```

* The singularity image should be pulled to the same directory that the code is cloned using: 
```    
    singularity pull <link is provided in the AD> 
```    
You can test the image by running the following command from the current directory:
```    
    singularity exec artifact.sif /source/fusion/build/demo/sptrsv_sptrsv_demo
```    
The output is a set of comma separated values such as specifications for a random matrix and execution time of different tools for the matrix.

* The datasets should be downloaded by calling:
```    
    python ssgetpy/dl_matrices.py 
```    
Matrices are downloaded into the _mm_ directory in the current directory (This might take a few hours and requires internet connection).

* The sparse fusion (Experiment 1) experiment can be executed by emitting:
```
	bash run_sfusion.sh
```
For running on compute node:
```
	sbatch bash run_sfusion.sh
```
You might need to update scripts with new absolute paths to the dataset and the image file. Also singularity module should be loaded for running on a server.


* The MKL evaluation (Experiment 2) can be done by running:
```
	bash run_mkl.sh
```    

* The DAGP evaluation (Experiment 3) can be reproduced by calling:
```
bash run_dagp.sh
``` 


* The Gauss-Seidel case study (Experiment 4) can be reproduced by calling:
```
	bash run_gs.sh
```
    
* Upon successful completion of experiments, all results should be stored as comma separated values (CSV) files under the _./logs/_ directory and are ready to be plotted. Separated Python scripts are provided and called inside the bash files to create plots. Plots are stored in the current directory as PDF files.

