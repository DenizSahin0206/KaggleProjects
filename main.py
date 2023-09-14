from NeuralNetworkScratch import NeuralNetFromScratch
import pandas as pd
import numpy as np

train_df = pd.read_csv(
    r"/Users/teleradio/Downloads/digit-recognizer/train.csv"
)

test_df = pd.read_csv(
    r"/Users/teleradio/Downloads/digit-recognizer/test.csv"
)

# For linear algebra, make the data into a Numpy Array
train_data = np.array(train_df)
train_data_T = train_data.T

# Get dimensions of data: M number of datapoints (rows), N: number of variables/features (columns)
n, m = train_data_T.shape

# Define the variables X and Y
x_train_T = train_data_T[1:n]
y_train_T = train_data_T[0]


# Train model
nns = NeuralNetFromScratch()
model = nns.train_model(x_train_T, y_train_T)
