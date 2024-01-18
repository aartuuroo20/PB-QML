import tensorflow_quantum as tfq
import tensorflow as tf
import cirq
import sympy
import random
import matplotlib.pyplot as plt
import numpy as np

from Circuit import Circuit
from DataSet import DataSet
from qat.interop.cirq import qlm_to_cirq


def make_data(n1, n2):
    qubit = cirq.GridQubit(0,0)
    train, test = [], []
    train_label, test_label = [], []
    for _ in range(n1):
        cir = cirq.Circuit()
        rot = random.uniform(0,0.1) if random.random() < 0.5 else random.uniform(0.9,1)
        cir.append([cirq.X(qubit)**rot])
        train.append(cir)
        if rot < 0.5:
            train_label.append(1)
        else:
            train_label.append(-1)
    for _ in range(n2):
        cir = cirq.Circuit()
        rot = random.uniform(0,0.1) if random.random() < 0.5 else random.uniform(0.9,1)
        cir.append([cirq.X(qubit)**rot])
        test.append(cir)
        if rot < 0.5:
            test_label.append(1)
        else:
            test_label.append(-1)
    return tfq.convert_to_tensor(train), np.array(train_label), tfq.convert_to_tensor(test), np.array(test_label)


def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)



dataset = DataSet()
dataset.Draw()

qubit = cirq.GridQubit(0,0)

#Instaciamos la clase circuito donde creamos un circuito cuantico y añadimos el encoding y el circuito variacional
circuit = Circuit()
circuit.varCircuit1()
circuit.ZZFeatureMap()
cirq_circuit = qlm_to_cirq(circuit.circuit())

no_measurements_circuit = cirq.drop_terminal_measurements(cirq_circuit)
qubit_map =  {cirq.LineQubit(1): cirq.GridQubit(0,0), cirq.LineQubit(2): cirq.GridQubit(0,1)}
modified_circuit = no_measurements_circuit.transform_qubits(qubit_map = qubit_map)
print(modified_circuit)

train, train_label, test, test_label = make_data(1000, 100)

readout_operators = [cirq.X(qubit)]
inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)

layer1 = tfq.layers.PQC(modified_circuit, readout_operators, repetitions=32, differentiator=tfq.differentiators.ParameterShift(), initializer=tf.keras.initializers.Zeros)(inputs)

model = tf.keras.models.Model(inputs=inputs, outputs=layer1)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.hinge, metrics=[hinge_accuracy])

#history = model.fit(train, train_label, epochs=64, batch_size=32, validation_data=(test, test_label))
history = model.fit(dataset.X_aux, epochs=64, batch_size=32, validation_split=0.1)

print(model.trainable_weights)

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