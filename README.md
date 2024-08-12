# Final Degree Project

This repository contains the project developed for the final degree work in computer engineering. 

The project consists in the development of a benchmarking platform for quantum machine learning called PB-ACC and evaluating the current tools for training quantum models using the interoperability module of myQLM. The frameworks to be evaluated are Qiskit, Cirq and pyQuil. Each framework is developed in a virtual environment independent of each other. 

The operation consists in the development of a series of variational circuits that allow the training of quantum models and a data encoding circuit in myQLM. Subsequently, using the interoperability module, the circuits are interoperated and implemented in each framework for subsequent training, evaluation and comparison.

During the development a problem arose, it was not possible to perform a correct interoperability between myQLM and pyQuil so Eviden support was contacted and the problem was communicated. The bug will be fixed in future releases. 

 ## Installation

  ### myQLM

 1. Install Python in version 3.9.
 2. Install myQLM.
 ```bash
pip install myqlm==1.9.9
```

 ### Qiskit

 1. Install Python in version 3.9.13.
 2. Install Qiskit.
  ```bash
pip install qiskit==0.43.1
```
 4. Install Qiskit machine learning.
 ```bash
pip install qiksit-machinelearning==0.7.2 
```
 6. To enable interoperability between myQLM and Qiskit it is necessary to install myqlm-interop[qiskit_binder].
 ```bash
pip install myqlm-interop[qiskit_binder]
```

### Cirq

 1. Download miniconda.
  ```bash
curl https://repo.anaconda.com/miniconda/Miniconda3-py39-23.5.2-0-Linuxx86-64.sh -o Miniconda3-py39-23.5.2-0-Linux-x86-64.sh 
```
 2. Install miniconda.
  ```bash
bash Miniconda3-py39-23.5.2-0-Linux-x86-64.sh
```
 4. Create a virtual environment in python version 3.8.
 5. Activate the virtual environment and install cuda tool kit.
 ```bash
conda install -c conda-forge cudatoolkit=11.8.0
```
 7. Install library cuDNN of NVIDIA.
 ```bash
pip install nvidia-cudnn-cu11==8.6.0.163
```
 9. Create a directory and a script file to set environment variables related to cuDNN.
 ```bash
mkdir -p $CONDA-PREFIX/etc/conda/activate.d 
```
 11. Add a line to the file.
 ```bash
echo ’CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.file)"))’ » $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
 13. Set an environment variable.
 ```bash
echo ’export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNN_PATH/lib:$LD_LIBRARY_PATH’ » $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
 15. Install tensorflow.
 ```bash
pip install tensorflow==2.7.*
```
 17. Install tensorflow quantum.
 ```bash
pip3 install -U tensorflow-quantum
```
 19. To enable interoperability between myQLM and Cirq it is necessary to install myqlm-interop[cirq_binder].
```bash
pip install myqlm-interop[cirq_binder]
```

### pyQuil & Pennylane

 1. Install Python in version 3.9.
 2. Install pyQuil
  ```bash
pip install pyquil
```
 4. Once pyQuil is installed, it is necessary to install a series of dependencies: https://docs.rigetti.com/qcs/getting-started/set-up-your-environment/installing-locally
 5. Install Pennylane
  ```bash
pip install pennylane
```
 7. To enable interoperability between pennylane and pyQuil it is necessary to install the following library: pennylane-rigetti
```bash
python -m pip install pennylane-rigetti
```

