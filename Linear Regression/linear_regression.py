import numpy
import pandas
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = (12.0, 9.0)

data = pandas.read_csv('data.csv')
X = data.iloc[:,0]
Y = data.iloc[:,1]
I = []
J = []
plt.scatter(X, Y)

slope = 0
intercept = 0
learning_rate = 0.0001
max_epochs = 1000
N = float(len(X))

# finding the slope and intercept
for i in range(max_epochs):
    #predicting Y using current slope and intercept
    Y_pred = slope * X + intercept

    #partial derivatives of J with respect to slope and intercept
    J_intercept = (2/N) * sum(Y_pred - Y)
    J_slope = (2/N) * sum((Y_pred - Y) * X)
    
    #updating slope and intercept by subtracting (learning rate * partial derivative) from the current values
    intercept -= learning_rate * J_intercept
    slope -= learning_rate * J_slope

    #calculating loss/cost, this is the value that should be minimized
    if i % 1 == 0:
        I.append(i)
        J.append((1/N) * sum((Y - Y_pred)*(Y - Y_pred)))


#modeling the found slope and intercept
Y_pred = slope*X + intercept

#plotting the modeled values
plt.scatter(X,Y)
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='blue')
plt.show()

#plotting the loss values (for every 100 iterations)
plt.scatter(I, J)
plt.show()
