import matplotlib as plt 
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score 

data =pd.read_csv("/Users/pranavranjan/Desktop/prodigy/House Price Prediction Dataset.csv")
df = pd.DataFrame(data)
x = df[['Area','Bedrooms','Bathrooms']]
y = df['Price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 42) 

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("Coefficients:",model.coef_)
print("Intercept",model.intercept_)
print("Mean Squared Error:",mse)
print("R2 score:",r2)


new_house = pd.DataFrame([[2200, 3, 2]], columns=['Area','Bedrooms','Bathrooms'])
predicted_price = model.predict(new_house)
print("predicted  price for new house:",predicted_price[0])

