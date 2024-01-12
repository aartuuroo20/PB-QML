import cirq
import sympy

from qat.interop.cirq import cirq_to_qlm
from qat.interop.cirq import qlm_to_cirq

from Test1 import Circuit


circuit = Circuit()
circuit.testCircuit()
cirq_circuit = qlm_to_cirq(circuit.circuit())
qubit_map =  {cirq.LineQubit(1): cirq.GridQubit(0,0)}

no_measurements_circuit = cirq.drop_terminal_measurements(cirq_circuit)
modified_circuit = no_measurements_circuit.transform_qubits(qubit_map = qubit_map)


print(modified_circuit)
