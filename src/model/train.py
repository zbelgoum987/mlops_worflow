import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import argparse
import glob
import mlflow 

def read_data(path):
    csv_files = glob.glob(f"{path}/*.csv") 
    if not csv_files: 
        raise RuntimeError(f"No CSV files found in provided data path: {path}") 
    df = pd.concat((pd.read_csv(f) for f in csv_files), sort=False)
    return df


def preprocess_data(df):
    X,y = df[df.columns[1:-1]].values, df[df.columns[-1]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def train_and_test_model(X_train, X_test, y_train, y_test, reg_rate):
    model = LogisticRegression(C=1/reg_rate, random_state=0).fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    print(f"training accuracy: {train_acc}")
    mlflow.log_metric("training accuracy", train_acc)
    test_acc = model.score(X_test, y_test)
    print(f"testing accuracy: {test_acc}")
    mlflow.log_metric("testing accuracy", test_acc)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01)
    args = parser.parse_args()
    return args


def main(args):
    mlflow.autolog()
    df = read_data(args.training_data)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    train_and_test_model(X_train, X_test, y_train, y_test, args.reg_rate)


if __name__ == "__main__":
    args = parse_args()
    main(args)



