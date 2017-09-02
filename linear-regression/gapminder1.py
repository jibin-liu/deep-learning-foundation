# TODO: Add import statements
from sklearn import linear_model
import pandas as pd

# Assign the dataframe to this variable.
# TODO: Load the data
bmi_life_data = pd.read_csv(r'./data/bmi_and_life_expectancy.csv')
bmi = bmi_life_data[['BMI']]
life_expectancy = bmi_life_data[['Life expectancy']]

# Make and fit the linear regression model
# TODO: Fit the model and Assign it to bmi_life_model
bmi_life_model = linear_model.LinearRegression()
bmi_life_model.fit(bmi, life_expectancy)

# Mak a prediction using the model
# TODO: Predict life expectancy for a BMI value of 21.07931
laos_life_exp = bmi_life_model.predict([21.07931])
print laos_life_exp
