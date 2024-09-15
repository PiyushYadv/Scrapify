import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('scraps_prices.csv')

label_encoder = LabelEncoder()
data['scrap_type_encoded'] = label_encoder.fit_transform(data['scrap_type'])

data.dropna(inplace=True)


X = data[['scrap_type_encoded', 'quantity', 'region', 'external_factor_1', 'external_factor_2']]  # Add all relevant features
y = data['price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


new_scrap_data = pd.DataFrame({
    'scrap_type_encoded': [label_encoder.transform(['metal'])[0]],
    'quantity': [100],
    'region': [2],
    'external_factor_1': [1200],
    'external_factor_2': [3]
})

predicted_price = model.predict(new_scrap_data)
print(f"Predicted Price: {predicted_price[0]}")
