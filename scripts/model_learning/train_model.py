import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

train = pd.read_csv('../../data/processed/train_final.csv')

preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['CountryCode', 'Status', 'ReactorType', 'ReactorModel'])
])

model = RandomForestRegressor(n_estimators=100, random_state=0)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

X = train.drop('Capacity', axis=1)
y = train['Capacity']

pipeline.fit(X, y)

with open('../../models/model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
