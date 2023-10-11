import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import pickle

test = pd.read_csv('../data/processed/test_final.csv')

with open('../models/model.pkl', 'rb') as f:
    model = pickle.load(f)

X_test = test.drop('Capacity', axis=1)
y_test = test['Capacity']

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

metrics = {
    'MAE': mae,
    'MSE': mse,
    'R2': r2,
}

with open('../metrics/metrics.json', 'w') as f:
    json.dump(metrics, f)
