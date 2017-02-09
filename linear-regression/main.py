import numpy as np
import matplotlib.pyplot as plt

def compute_error(b, m, pts):
    return sum((pts[:, 1] - (m*pts[:, 0] + b)) ** 2) / float(len(pts))

def update(b, m, pts, learning_rate):
    b_grad = 0
    m_grad = 0
    N = float(len(pts))

    # Calculate the partial derivatives.
    for i in range(0, len(pts)):
        x = pts[i, 0]
        y = pts[i, 1]
        b_grad += -(2/N) * (y - (m*x + b))
        m_grad += -(2/N) * x * (y - (m*x + b))

    # Update b and m using these partial derivatives.
    b_new = b - (learning_rate*b_grad)
    m_new = m - (learning_rate*m_grad)

    return [b_new, m_new]

def gradient_descent(pts, b, m, learning_rate, num_iter):
    for i in range(0, num_iter):
        # Update the values of b and m.
        b, m = update(b, m, pts, learning_rate)

        # Display the results after every 100 iterations.
        if (i % 100 == 0):
            print('After iteration #%d: b=%f, m=%f, error=%f' % \
                (i, b, m, compute_error(b, m, pts)))

    return [b, m]

def run():
    """
    Read the data from a csv file. The data set is a number of values,
    separated in two columns. Column 0 is the amount of hours studied
    by students in an university/college, and column 1 is the scores
    they obtained in the tests. Highly relatable and authentic.
    """
    data_points = np.genfromtxt('data.csv', delimiter=',')

    """
    Define the hyperparameters for the machine learning model.
    """
    # How fast should the model converge?
    learning_rate = 0.0001

    # Define the initial values for b and m. b and m being the y-intercept
    # and the slope of the fit line, respectively.
    init_b = 0
    init_m = 0

    # Number of iterations.
    num_iter = 1000

    """
    Train the damn model.
    """
    print('Starting gradient descent at b=%f, m=%f, error=%f' % \
        (init_b, init_m, compute_error(init_b, init_m, data_points)))

    [b, m] = gradient_descent(data_points, init_b, init_m, learning_rate, num_iter)
    print('Finished gradient descent at b=%f, m=%f, error=%f' % \
        (b, m, compute_error(b, m, data_points)))

    """
    Plot the data points and the line that best fits the data.
    """
    line_values = np.array(m*data_points[:, 0] + b)
    plt.plot(data_points[:, 0], data_points[:, 1], 'ro')
    plt.plot(data_points[:, 0], line_values, 'b')
    plt.ylabel('test scores')
    plt.xlabel('amount of hours studied')
    plt.title('Linear Regression demo')
    plt.show()

if __name__ == '__main__':
    run()
