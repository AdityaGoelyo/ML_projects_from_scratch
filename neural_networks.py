import numpy as np

# classes of network related objects

class perceptron():
    # an object that can store weights and do a forward pass
    def __init__(self, dimension):
        self.weights = np.random.randn(dimension)

    def forward_pass(self, input):
        return np.dot(input, self.weights)

    
    def train_by_perceptron_training_rule(self, values, target, n):
        prediction = self.forward_pass(values)
        w_hat = np.mean(n*np.dot((target - prediction),values), axis=0)
        self.weights = self.weights + w_hat

    def train_by_gradient_descent_on_MSE(self, values, target, n):
        prediction = self.forward_pass(values)
        w_hat = np.mean(
            2*n*np.dot((target-prediction),values)
            )
        self.weights = self.weights + w_hat

# Cost Functions

def MSE(y_hat, y):
    return np.mean((y-y_hat)**2)

# sample example

o = perceptron(4)

target = np.random.randint(0, 2, size=(5))
values = np.random.randint(0, 2, size=(5, 4))

for i in range(1000):
    o.train_by_gradient_descent_on_MSE(values, target, 0.01)
    if i % 10 == 0:
        print(MSE(o.forward_pass(values), target))