# Final Degree Project

This repository contains the project developed for the final degree work in computer engineering. 

The project consists in the development of a benchmarking platform for quantum machine learning called PB-ACC and evaluating the current tools for training quantum models using the interoperability module of myQLM. The frameworks to be evaluated are Qiskit, Cirq and pyQuil. Each framework is developed in a virtual environment independent of each other. 

The operation consists in the development of a series of variational circuits that allow the training of quantum models and a data encoding circuit in myQLM. Subsequently, using the interoperability module, the circuits are interoperated and implemented in each framework for subsequent training, evaluation and comparison.

During the development a problem arose, it was not possible to perform a correct interoperability between myQLM and pyQuil so Eviden support was contacted and the problem was communicated. The bug will be fixed in future releases. 

 ## Installation

  ### myQLM

 1. Install Python in version 3.9
 2. Install myQLM
 ```bash
pip install myqlm==1.9.9
```

 ### Qiskit

 1. Install Python in version 3.9.13
 2. Install Qiskit in version 0.43.1
 3. Install Qiskit machine learning in version 0.7.2
 4. To enable interoperability between myQLM and Qiskit it is necessary to install myqlm-interop[qiskit_binder].

### Cirq

 1. Download and install miniconda
 2. Create a virtual environment in python version 3.8
 3. Activate the virtual environment and install cuda tool kit in version 11.8.0
 4. Install library cuDNN of NVIDIA in version 8.6.0.163
 5. Create a directory and a script file to set environment variables related to cuDNN
 6. Add a line to the file: echo 'CUDNNN_PATH=$$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.file)")' " $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
 7. Set an environment variable: echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$CUDNNN_PATH/lib:$LD_LIBRARY_PATH' " $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
 8. Install tensorflow in version 2.7
 9. Install tensorflow quantum
 10. To enable interoperability between myQLM and Cirq it is necessary to install myqlm-interop[cirq_binder].

### pyQuil & Pennylane

 1. Install Python in version 3.9
 2. Install pyQuil
 3. Once pyQuil is installed, it is necessary to install a series of dependencies: https://docs.rigetti.com/qcs/getting-started/set-up-your-environment/installing-locally
 4. Install Pennylane
 5. To enable interoperability between pennylane and pyQuil it is necessary to install the following library: pennylane-rigetti

