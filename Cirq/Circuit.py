import numpy as np

from qat.lang import Program, H, PH, CNOT, qrout, RY, X, RZ
from qat.core import Variable

class Circuit:
    #Constructor that initialize the circuit with 2 qubits and create the qprogram
    def __init__(self):
        nqubits = 2
        self.qprogram = Program()
        self.qubits = self.qprogram.qalloc(nqubits)

    #Function that create the ZZFeatureMap and apply it to the qprogram
    def ZZFeatureMap(self):
        ListVarData = []
        for i in range(2):
            ListVarData.append(Variable("varData" + str(i)))
        
        varaibleAux = Variable("varAux")
        
        alpha = ListVarData[0] * ListVarData[1]
        a = [0,1]
        b = [1,1]

        @qrout
        def zzmf():
            H(0)
            H(1)
            PH(2* ListVarData[0])(0)
            PH(2* ListVarData[1])(0)
            CNOT(0, 1)
            #PH(2*(np.pi - ListVarData[0]) * (np.pi - ListVarData[1]))(1)
            PH(np.pi**2)(1)
            PH(- np.pi * ListVarData[1])(1)
            PH(- np.pi * ListVarData[0])(1)
            PH(varaibleAux * ListVarData[0])(1)

            CNOT(0, 1)
            H(0)
            H(1)
            PH(2* ListVarData[0])(0)
            PH(2* ListVarData[1])(0)
            CNOT(0, 1)
            #PH(2*(np.pi - ListVarData[0]) * (np.pi - ListVarData[1]))(1)
            CNOT(0, 1)
        
        self.qprogram.apply(zzmf, self.qubits)

    #Function that create the variational circuit and apply it to the qprogram
    def varCircuit1(self):
        ListVarTheta = []
        for i in range(8):
            ListVarTheta.append(Variable("varTheta" + str(i)))

        @qrout
        def varcircuit():
            RY(ListVarTheta[0])(0)
            RY(ListVarTheta[1])(1)
            CNOT(0, 1)
            RY(ListVarTheta[2])(0)
            RY(ListVarTheta[3])(1)
            CNOT(0, 1)
            RY(ListVarTheta[4])(0)
            RY(ListVarTheta[5])(0)
            CNOT(0, 1)
            RY(ListVarTheta[6])(0)
            RY(ListVarTheta[7])(1)
        
        self.qprogram.apply(varcircuit, self.qubits)

    def varCircuit2(self):
        ListVarTheta = []
        for i in range(5):
            ListVarTheta.append(Variable("varTheta" + str(i))) 

        @qrout
        def varcircuit():
            RY(ListVarTheta[0])(0)
            CNOT(0, 1)
            RY(ListVarTheta[1])(1)
            CNOT(0, 1)
            RY(ListVarTheta[2])(1)
            X(0)
            CNOT(0, 1)
            RY(ListVarTheta[3])(1)
            CNOT(0, 1)
            RY(ListVarTheta[4])(1)
            X(0)
        
        self.qprogram.apply(varcircuit, self.qubits)
    
    def varCircuit3(self):
        ListVarTheta = []
        for i in range(2):
            ListVarTheta.append(Variable("varTheta" + str(i))) 

        @qrout
        def varcircuit():
            H(0)
            CNOT(0, 1)
            RZ(ListVarTheta[0])(0)
            RZ(ListVarTheta[1])(1)
            CNOT(0, 1)
            H(0)
        
        self.qprogram.apply(varcircuit, self.qubits)

    #Function that display the circuit
    def display(self):
        circuit = self.qprogram.to_circ()
        circuit.display()

    #Function that return the circuit
    def circuit(self):
        return self.qprogram.to_circ()