# Linear Regression using Gradient Descent
# First attempt

import numpy as np
import matplotlib .pyplot as plt

# Load data
data = np.loadtxt('ex1data1.txt', delimiter=',')
population = data[:, 0]
profit = data[:, 1]

# Initialize values
theta0 = 0
theta1 = 0
n_samples = len(data)
learning_rate = 0.01
num_iterations = 1500


# Hypothesis Function: linear equation for the line fitted in the data for prediction
def hypothesis(theta0, theta1, population):
    profit_hat = theta0 + (theta1 * population)
    return profit_hat


# Cost Function (aka Sum of Squared Errors): checks the accuracy of the fitted line
def cost_function(profit, n_samples, theta0, theta1):
    error = profit - hypothesis(theta0, theta1, population)
    sq_error = error ** 2
    sum_sq_error = np.sum(sq_error)
    cost = 1/(2*n_samples) * sum_sq_error
    return cost


# Plotting line to data
def plot_line(population, profit, theta0, theta1):
    predict_hat = hypothesis(theta0, theta1, population)

    plt.scatter(population, profit, c='red', marker='x')
    plt.plot([min(population), max(population)], [min(predict_hat), max(predict_hat)], c='blue')
    plt.xlabel('Population per City in $10,000s')
    plt.ylabel('Profit per City in 10,000s')
    plt.show()


# Gradient Descent
def gradient_descent(theta0, theta1, n_samples, learning_rate, num_iterations):
    # Updated values of thetas for each iteration
    updated_theta0 = theta0
    updated_theta1 = theta1

    for i in range(num_iterations):
        # Partial derivatives of Cost Function with respect to theta0 and theta1
        gradient0 = 1/n_samples * np.sum(hypothesis(updated_theta0, updated_theta1, population) - profit)
        gradient1 = 1/n_samples * np.sum((hypothesis(updated_theta0, updated_theta1, population) - profit) * population)

        updated_theta0 = updated_theta0 - learning_rate * gradient0
        updated_theta1 = updated_theta1 - learning_rate * gradient1

    plot_line(population, profit, updated_theta0, updated_theta1)

    print(f"After Gradient Descent, the following theta values were obtained: \nTheta 0: {updated_theta0}\nTheta 1: {updated_theta1}")

if __name__ == '__main__':
    gradient_descent(theta0, theta1, n_samples, learning_rate, num_iterations)
