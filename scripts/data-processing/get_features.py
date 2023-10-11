import pandas as pd
train_data_path = '../../data/raw/train.csv'
test_data_path = '../../data/raw/test.csv'

train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

features = ['Latitude', 'Longitude', 'CountryCode', 'Status', 'ReactorType', 'ReactorModel', 'ConstructionStartAt', 'OperationalFrom']

train_df = train_df[features + ['Capacity']]
test_df = test_df[features + ['Capacity']]

train_save_path = '../../data/processed/train_processed.csv'
test_save_path = '../../data/processed/test_processed.csv'

train_df.to_csv(train_save_path, index=False)
test_df.to_csv(test_save_path, index=False)
