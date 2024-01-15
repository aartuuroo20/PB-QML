import numpy as np

from qat.lang import Program, H, PH, CNOT, qrout, RY, X, RZ, RX
from qat.core import Variable

class Circuit:
    #Constructor that initialize the circuit with 2 qubits and create the qprogram
    def __init__(self):
        nqubits = 2
        self.qprogram = Program()
        self.qubits = self.qprogram.qalloc(nqubits)
    
    def testCircuit(self):
        ListVarTheta = []
        for i in range(3):
            ListVarTheta.append(Variable("varTheta" + str(i))) 

        @qrout
        def varcircuit():
            RX(ListVarTheta[0])(0)
            RY(ListVarTheta[1])(0)
            RZ(ListVarTheta[2])(0)

            RX(ListVarTheta[0])(1)
            RY(ListVarTheta[1])(1)
            RZ(ListVarTheta[2])(1)

        self.qprogram.apply(varcircuit, self.qubits)

    #Function that display the circuit
    def display(self):
        circuit = self.qprogram.to_circ()
        circuit.display()

    #Function that return the circuit
    def circuit(self):
        return self.qprogram.to_circ()