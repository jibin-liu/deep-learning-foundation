import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
import os


def get_data(filename):
    return os.path.join(os.getcwd(), 'data', filename)


# read body weight data
df = pd.read_fwf(get_data('brain_body.txt'))
# it has to use df[['Brain']] here to get a 2d data of x_values
# if use df['Brain'], x/y_values will be 1d data, and cannot be used in LR model below.
x_values = df[['Brain']]
y_values = df[['Body']]

# train the linear regression model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

# plot raw data and the regression
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
