import numpy as np
import matplotlib.pyplot as plt

class DataSet:
    #Constructor
    def __init__(self, seed=None):
        self.seed = seed
        self.CreateDataSet()

    #Function that create a dataset of 20 samples with 2 inputs   
    def CreateDataSet(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            
        num_samples = 50
        num_inputs = 2

        X_aux = 2 * np.random.rand(num_samples, num_inputs) - 1
        y01 = 1 * (np.sum(X_aux, axis=1) >= 0)
        y = 2 * y01 - 1  

        self.X_aux = X_aux
        self.y = y

    #Function that return the dataset
    def GetItems(self):
        print(self.X_aux)
    
    #Function that create a plot of the dataset
    def Draw(self):
        for x, y_target in zip(self.X_aux, self.y):
            if y_target == 1:
                plt.plot(x[0], x[1], "bo")
            else:
                plt.plot(x[0], x[1], "go")
        plt.plot([-1, 1], [1, -1], "--", color="black")
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Clasificación Binaria Cuántica en Qiskit')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close()