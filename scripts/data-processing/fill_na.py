import pandas as pd
train_processed_data_path = '../../data/processed/train_processed.csv'
test_processed_data_path = '../../data/processed/test_processed.csv'

train_df = pd.read_csv(train_processed_data_path)
test_df = pd.read_csv(test_processed_data_path)

train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

train_save_path = '../../data/processed/train_final.csv'
test_save_path = '../../data/processed/test_final.csv'

train_df.to_csv(train_save_path, index=False)
test_df.to_csv(test_save_path, index=False)
