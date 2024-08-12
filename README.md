# Final Degree Project

This repository contains the project developed for the final degree work in computer engineering. 

The project consists in the development of a benchmarking platform for quantum machine learning called PB-ACC. This project consists of evaluating the current tools for training quantum models using the interoperability module of myQLM. The frameworks to be evaluated are Qiskit, Cirq and pyQuil. Each framework is developed in a virtual environment independent of each other. 

The operation consists in the development of a series of variational circuits that allow the training of quantum models and a data encoding circuit in myQLM. Subsequently, using the interoperability module, the circuits are interoperated and implemented in each framework for subsequent training, evaluation and comparison.

 ## Installation

 ### Qiskit

 1. Install Python in version 3.9.13
 2. Install Qiskit in version 0.43.1
 3. Install Qiskit machine learning in version 0.7.2
 4. To enable interoperability between myQLM and Qiskit it is necessary to install myqlm-interop[qiskit_binder].
