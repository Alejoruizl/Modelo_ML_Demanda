import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestClassifier

# Cargar los datos
demand_data = pd.read_csv('dataset_demand_acumulate.csv')
alpha_beta_data = pd.read_csv('dataset_alpha_betha.csv')
to_predict_data = pd.read_csv('to_predict.csv')

# Análisis estadístico
print(demand_data.describe())

# Visualización de la demanda histórica
print(demand_data.columns)
plt.figure(figsize=(10,6))
plt.plot(demand_data['Date'], demand_data['Demand'], label='Demanda')
plt.xlabel('Fecha')
plt.ylabel('Demanda')
plt.title('Demanda Histórica de Cementos Argos')
plt.legend()
plt.show()

demand_data['Date'] = pd.to_datetime(demand_data['Date'])
train_data = demand_data[demand_data['Date'] < '2022-01-01']
test_data = demand_data[demand_data['Date'] >= '2022-01-01']

# Modelado de la demanda
model = ExponentialSmoothing(train_data['Demand'], seasonal='add', seasonal_periods=12).fit()
demand_forecast = model.forecast(steps=7)

# Añadir los pronósticos al dataframe de prueba
test_data['Forecast'] = demand_forecast.values

# Evaluación del modelo
mse = mean_squared_error(test_data['Demand'], test_data['Forecast'])
r2 = r2_score(test_data['Demand'], test_data['Forecast'])

print(f'MSE: {mse}, R2: {r2}')

plt.figure(figsize=(10,6))
plt.plot(train_data['Date'], train_data['Demand'], label='Datos de Entrenamiento')
plt.plot(test_data['Date'], test_data['Demand'], label='Datos de Validación')
plt.plot(test_data['Date'], test_data['Forecast'], label='Pronóstico')
plt.xlabel('Fecha')
plt.ylabel('Demanda')
plt.title('Predicción de la Demanda de Cementos Argos')
plt.legend()
plt.show()


# Preparar los datos
X = alpha_beta_data.drop(columns=['Class'])
y = alpha_beta_data['Class']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Clasificar los nuevos datos
predictions = classifier.predict(to_predict_data.drop(columns=['Demand', 'Class']))
to_predict_data['Class'] = predictions

# Guardar el resultado
to_predict_data.to_csv('predicted_classes.csv', index=False)