import numpy as np

def sigmoid(t, deriv = False):              # The Sigmoid function
    if(deriv == True):
        return t * (1 - t)

    return 1 / (1 + np.exp(-t))

X = np.array( [ [0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1] ] )  # Input
Y = np.array( [ [0], [1], [1], [0] ] )                          # Output

np.random.seed(1)

syn0 = 2 * np.random.random((3, 1)) - 1     # Synapse 0

for i in range(60000):
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))

    l1_del = (Y - l1) * sigmoid(l1, True)   # Delta

    syn0 += np.dot(l0.T, l1_del)            # Update weights

print("Output after training")
print(l1)

