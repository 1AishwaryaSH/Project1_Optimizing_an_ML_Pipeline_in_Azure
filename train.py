from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

data_path= "https://raw.githubusercontent.com/1AishwaryaSH/ML-Engineer-with-Azure-Capstone/main/Dataset_spine.csv"
ds = TabularDatasetFactory.from_delimited_files(path = data_path)



def clean_data(data):
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    x = x_df.drop(columns = ['Class_att'])
    y_df = x_df.pop("Class_att").apply(lambda s: 0 if s == "Abnormal" else 1) 
    return x, y_df

x, y = clean_data(ds)
y

### YOUR CODE HERE ###a
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)



run = Run.get_context()



def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()
