import numpy as np

def sigmoid(t, deriv = False):              # The Sigmoid function
    if(deriv == True):
        return t * (1 - t)

    return 1 / (1 + np.exp(-t))

X = np.array( [ [0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1] ] )  # Input
Y = np.array( [ [0], [1], [1], [0] ] )                          # Output

np.random.seed(1)

syn0 = 2 * np.random.random((3, 4)) - 1     # Synapse 0
syn1 = 2 * np.random.random((4, 1)) - 1     # Synapse 1

for i in range(60000):
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))

    if(i % 10000 == 0):
        print("Error:" + str(np.mean(np.abs(Y - l2))))

    l2_del = (Y - l2) * sigmoid(l2, True)
    l1_del = np.dot(l2_del, syn1.T) * sigmoid(l1, True)

    syn1 += np.dot(l1.T, l2_del)
    syn0 += np.dot(l0.T, l1_del)

print("Output after training")
print(l2)
