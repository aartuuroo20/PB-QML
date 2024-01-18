import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow_quantum as tfq
import tensorflow as tf
import cirq
import sympy

from DataSet import DataSet

def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)

class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout)**symbol)

def create_quantum_model(num_layers=1):
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(4, 4)  # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()
    
    builder = CircuitLayerBuilder(
        data_qubits = data_qubits,
        readout=readout)

    for i in range(num_layers):
        builder.add_layer(circuit, cirq.ZZ, "zz1{}".format(i + 1))
        builder.add_layer(circuit, cirq.XX, "xx{}".format(i + 1))

    return circuit, cirq.Z(readout)

def make_circuit(qubit):
    x = sympy.symbols('X_rot')
    y = sympy.symbols('Y_rot')
    z = sympy.symbols('Z_rot')
    c = cirq.Circuit()
    c.append(cirq.rx(x).on(qubit))
    c.append(cirq.ry(y).on(qubit))
    c.append(cirq.rz(z).on(qubit))
    return c


########################################################################################################################33

dataset = DataSet()

X_train, X_test, y_train, y_test = train_test_split(dataset.get_data(), dataset.get_labels(), test_size=0.2, random_state=42)
print('X_train:',np.shape(X_train))
print('y_train:',np.shape(y_train))
print('X_test:',np.shape(X_test))
print('y_test:',np.shape(y_test))

########################################################################################################################33

circuit = make_circuit(cirq.GridQubit(0,0))
readout = cirq.X(cirq.GridQubit(0,0))

inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)

########################################################################################################################33

'''
layer = tfq.layers.PQC(circuit, readout)(inputs)

out = (layer + 1) / 2
model = tf.keras.Model(inputs=inputs, outputs=out)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
                loss=tf.keras.losses.hinge,
                metrics=[hinge_accuracy])

history = model.fit(
      X_train_strings, y_train,
      batch_size=9,
      epochs=3,
      verbose=1,
      validation_data=(X_test_strings, y_test))

'''

layer1 = tfq.layers.PQC(circuit, readout, repetitions=32, differentiator=tfq.differentiators.ParameterShift(), initializer=tf.keras.initializers.Zeros)(inputs)
model = tf.keras.models.Model(inputs=inputs, outputs=layer1)
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=tf.keras.losses.hinge, metrics=[hinge_accuracy])

history = model.fit(X_train, y_train, epochs=64, batch_size=32, validation_data=(X_test, y_test))

########################################################################################################################33

plt.plot(history.history['loss'], label='Training')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Hinge Loss")
plt.show()

plt.plot(history.history['hinge_accuracy'], label='Training')
plt.plot(history.history['val_hinge_accuracy'], label='Validation Acc')
plt.legend()
plt.title("Training Acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()