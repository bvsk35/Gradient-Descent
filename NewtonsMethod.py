# Finding the minimum for the function:
# f(x,y) = -log(1-x-y)-log(x)-log(y)
# Domain x + y < 1, x > 0, y > 0
# Using Newton's Method

# Import Required Libraries
import numpy
import matplotlib.pyplot as plt

# Functions
# Initial Guess
def Initalisation():
    x = numpy.random.uniform(0, 0.5)
    y = numpy.random.uniform(0, 0.5)
    return numpy.array([[x, y]])

# Calculates Gradient of the given Function
def Gradient(x, y):
    g1 = (-2 * x - y + 1) / (x ** 2 + y * x - x)
    g2 = (-x - 2 * y + 1) / (x * y + y ** 2 - 1)
    return numpy.array([g1, g2])

# Calculates Hessian Matrix of the given Function
def Hessian(x, y):
    h11 = (1/x**2)+(1/(-y-x+1)**2)
    h12 = (1/(-y-x+1)**2)
    h21 = (1/(-y-x+1)**2)
    h22 = (1/y**2)+(1/(-y-x+1)**2)
    return numpy.array([[h11, h12], [h21, h22]])

# Evaluates the value of the function at the given point
def Eval_Func(x, y):
    return -numpy.log(1-x-y)-numpy.log(x)-numpy.log(y)

# Update the points based on the Gradient Descent Algorithm
def Update_Weights(x, y, eta):
    return numpy.array([x, y]) - eta*numpy.dot(numpy.linalg.inv(Hessian(x, y)), Gradient(x, y))

# Actual Loop where Gradient Descent Algo runs until optimal point is reached
def NM(max_iter, tol, eta):
    iterations = 0
    F = numpy.array([])  # Stores the values of the function
    Epoch = numpy.array([])
    while iterations < max_iter:
        if iterations == 0:
            # Generate Initial Guess and Book Keeping
            if numpy.DataSource().exists('InitialGuess.txt'):
                W = numpy.loadtxt('InitialGuess.txt')  # Load the Initial Weights
            else:
                W = Initalisation()
                numpy.savetxt('InitialGuess.txt', W)  # Generate the Weights and save them
            W = numpy.reshape(W, (1, 2))
            f_temp = Eval_Func(W[-1, 0], W[-1, 1])
            F = numpy.concatenate((F, [f_temp]), axis=0)
            Epoch = numpy.concatenate((Epoch, [iterations]), axis=0)
            print('No. of Iterations: ', iterations, ' Points: ', W[-1], ' Function Value: ', F[-1], '\n')
            iterations += 1
        else:
            # Run The Gradient Descent Algorithm
            w_temp = Update_Weights(W[-1, 0], W[-1, 1], eta)
            f_temp = Eval_Func(W[-1, 0], W[-1, 1])
            # Book Keeping
            W = numpy.concatenate((W, [w_temp]), axis=0)
            F = numpy.concatenate((F, [f_temp]), axis=0)
            Epoch = numpy.concatenate((Epoch, [iterations]), axis=0)
            print('No. of Iterations: ', iterations, ' Points: ', W[-1], ' Function Value: ', F[-1], '\n')
            # Check for Close Weights
            if (W[-1] - W[-2]).all() < tol:
                print('Optimal Value Reached')
                break
            else:
                iterations += 1
    return Epoch, W, F


# Parameters
max_iter = 5000 # Maximum Iterations to reach the optimal value
tol = 1.0e-6 # Tolerance
eta = 0.05 # Learning Rate

# Main Loop
Epoch, W, F = NM(max_iter, tol, eta)

# Plot the results
# Plot 1
fig, ax1 = plt.subplots()
ax1.plot(W[:, 0], W[:, 1], 'rx:', label='Points After Every Iterations')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectory Followed by the Points at each Iteration')
plt.legend()
# Plot 2
fig, ax2 = plt.subplots()
ax2.plot(Epoch, F, 'b', label='Energy')
plt.xlabel('Epoch')
plt.ylabel('Function Value')
plt.title('Value of the Function -log(1-x-y)-log(x)-log(y) at each iteration')
fig.text(0.39, 0.03, r'Learning Rate $\eta = 0.05$', ha='center')
plt.legend()
plt.tight_layout()
plt.show()