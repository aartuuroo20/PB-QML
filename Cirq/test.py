import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow_quantum as tfq
import tensorflow as tf
import cirq
from sklearn.preprocessing import MinMaxScaler

from Circuit import Circuit
from qat.interop.cirq import qlm_to_cirq

def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)


df = pd.read_csv("/home/arturo/Downloads/heart.csv")
print(df.head())

target_column = "output"
numerical_column = df.columns.drop(target_column)
output_rows = df[target_column]
df.drop(target_column,axis=1,inplace=True)

scaler = MinMaxScaler()
scaler.fit(df)
t_df = scaler.transform(df)

X_train, X_test, y_train, y_test = train_test_split(t_df, output_rows, test_size=0.25, random_state=0)

print('X_train:',np.shape(X_train))
print('y_train:',np.shape(y_train))
print('X_test:',np.shape(X_test))
print('y_test:',np.shape(y_test))


qubit = cirq.GridQubit(0,0)

#Instaciamos la clase circuito donde creamos un circuito cuantico y añadimos el encoding y el circuito variacional
circuit = Circuit()
circuit.varCircuit1()
cirq_circuit = qlm_to_cirq(circuit.circuit())

no_measurements_circuit = cirq.drop_terminal_measurements(cirq_circuit)
qubit_map =  {cirq.LineQubit(1): cirq.GridQubit(0,0), cirq.LineQubit(2): cirq.GridQubit(0,1)}
modified_circuit = no_measurements_circuit.transform_qubits(qubit_map = qubit_map)
print(modified_circuit)

readout_operators = [cirq.X(qubit)]

# Convierte los circuitos cuánticos a tensores de tipo string
X_train_strings = tfq.convert_to_tensor([modified_circuit for _ in range(len(X_train))])
X_test_strings = tfq.convert_to_tensor([modified_circuit for _ in range(len(X_test))])
y_test_strings = tfq.convert_to_tensor([modified_circuit for _ in range(len(y_test))])
y_train_strings = tfq.convert_to_tensor([modified_circuit for _ in range(len(y_train))])

inputs = tf.keras.Input(shape=(), dtype=tf.dtypes.string)

layer1 = tfq.layers.PQC(modified_circuit, readout_operators, repetitions=32, 
                        differentiator=tfq.differentiators.ParameterShift(), 
                        initializer=tf.keras.initializers.Zeros)(inputs)

model = tf.keras.models.Model(inputs=inputs, outputs=layer1)
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss=tf.keras.losses.hinge,  metrics=[hinge_accuracy])

#history = model.fit(train, train_label, epochs=64, batch_size=32, validation_data=(test, test_label))
history = model.fit(X_train_strings, y_train, epochs=64, batch_size=32, validation_data=(X_test_strings, y_test))

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