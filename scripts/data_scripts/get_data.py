#!/usr/bin/python3

import pandas as pd
from sklearn.model_selection import train_test_split
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, '..', '..', 'data', 'raw', 'nuclear_power_plants.csv')
data = pd.read_csv(file_path)

train_data, test_data = train_test_split(data, test_size = 0.2, random_state=42)

raw_folder = os.path.join(current_dir, '..', '..', 'data', 'raw')
train_file_path = os.path.join(raw_folder, 'train.csv')
train_data.to_csv(train_file_path, index=False)

test_file_path = os.path.join(raw_folder, 'test.csv')
test_data.to_csv(test_file_path, index=False)
