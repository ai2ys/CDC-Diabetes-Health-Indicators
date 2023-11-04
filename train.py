import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
import logging
import os
import json

def preprocess_features(df, preproc_normalizers):
    df_preprocessed = df.copy(deep=True)

    # standard scaling (additional log for BMI)
    df_preprocessed.BMI = np.log(df_preprocessed.BMI)
    df_preprocessed.BMI = (
        df_preprocessed.BMI - preproc_normalizers['BMI']['mean']
        ) / preproc_normalizers['BMI']['std']
    df_preprocessed.Age = (
        df_preprocessed.Age - preproc_normalizers['Age']['mean']
        ) / preproc_normalizers['Age']['std']

    # Min max scaling
    df_preprocessed.MentHlth = (
        df_preprocessed.MentHlth - preproc_normalizers['MentHlth']['min']
        )/(preproc_normalizers['MentHlth']['max'] - preproc_normalizers['MentHlth']['min'])
    df_preprocessed.PhysHlth = (
        df_preprocessed.PhysHlth - preproc_normalizers['PhysHlth']['min']
        )/( preproc_normalizers['PhysHlth']['max'] - preproc_normalizers['PhysHlth']['min'] )
    df_preprocessed.GenHlth = (
        df_preprocessed.GenHlth - preproc_normalizers['GenHlth']['min']
        )/( preproc_normalizers['GenHlth']['max'] - preproc_normalizers['GenHlth']['min'])
    df_preprocessed.Education = (
        df_preprocessed.Education - preproc_normalizers['Education']['min']
        )/(preproc_normalizers['Education']['max'] - preproc_normalizers['Education']['min'])
    df_preprocessed.Income = (
        df_preprocessed.Income - preproc_normalizers['Income']['min']
        ) / ( preproc_normalizers['Income']['max'] - preproc_normalizers['Income']['min'] )

    #all other variables are binary, they do not need to be normalized

    return df_preprocessed

def load_full_train_test(path_train:str, path_val:str, path_test:str) -> (pd.DataFrame, pd.DataFrame):
    """Load the full train (train + validation) and test data

    Args:
        path_train (str): The path to the training data
        path_val (str): The path to the validation data
        path_test (str): The path to the test data

    Returns:
        (pd.DataFrame, pd.DataFRame): The full (train + val) training dataframe and test dataframe
    """
    df_train = pd.read_csv(path_train)
    df_val = pd.read_csv(path_val)
    df_test = pd.read_csv(path_test)

    df_train_full = pd.concat([df_train, df_val])
    return df_train_full, df_test

def preprocess_dataframe(df:pd.DataFrame, data_normalizers:dict) -> (pd.DataFrame, pd.Series):
    """Create X and y from the dataframe including extracting the target column and preprocessing the features.

    Args:
        df (pd.DataFrame): The dataframe to be preprocessed including the target column

    Returns:
        (pd.DataFrame, pd.Series): The preprocessed X data and y data
    """
    target_column = 'Diabetes_binary'
    # define the feature columns
    feature_columns = [col for col in df.columns if col != target_column]

    # split the data into train and val data
    X = preprocess_features(df[feature_columns], data_normalizers)
    y = df[target_column]
    return X, y

def print_metrics(y_true:np.array, y_pred:np.array) -> str:
    """Return the metrics of the model within a formatted string.

    Args:
        y_true (np.array): The true labels
        y_pred (np.array): The predicted labels

    Returns:
        str: The metrics of the model within a formatted string.
    """
    metrics_str = ""
    # overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    metrics_str += f"Overall accuracy: {accuracy:.4f}\n"
    accuracy_class_0 = accuracy_score(y_true[y_true==0], y_pred[y_true==0])
    accuracy_class_1 = accuracy_score(y_true[y_true==1], y_pred[y_true==1])
    metrics_str += f"Accuracy class 0: {accuracy_class_0:.4f}\n"
    metrics_str += f"Accuracy class 1: {accuracy_class_1:.4f}\n"
    metrics_str += "Classification report:\n"
    metrics_str += classification_report(y_true, y_pred)
    return metrics_str


class TrainRandomForest():
    def __init__(
            self, 
            hparams:dict, 
            data_normalizers:dict, 
            seed:int, 
            path_train:str, 
            path_val:str, 
            path_test:str, ):
        """Create a TrainRandomForest model.

        Args:
            hparams (dict): The hyperparameters of the model
            seed (int): The seed
            path_train (str): The path to the training data
            path_val (str): The path to the validation data
            path_test (str): The path to the test data
        """
        self.hparams = hparams
        self.data_normalizers = data_normalizers
        self.seed = seed
        self.df_train_full, self.df_test = load_full_train_test(
            path_train, path_val, path_test)
        self.X_train_full, self.y_train_full = preprocess_dataframe(self.df_train_full, self.data_normalizers)
        self.X_test, self.y_test = preprocess_dataframe(self.df_train_full, self.data_normalizers)

        self.model = RandomForestClassifier(
            random_state=self.seed, **self.hparams, class_weight='balanced', n_jobs=-1)

        self.dv = DictVectorizer(sparse=False)
        self.X_train_full = self.dv.fit_transform(self.X_train_full.to_dict(orient='records'))
        self.X_test = self.dv.transform(self.X_test.to_dict(orient='records'))

    def train(self):
        self.model.fit(self.X_train_full, self.y_train_full)


    def predict(self, X):
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        return y_pred, y_pred_proba

    def save_model(self, filename_model:str='model_random_forest.bin', filename_dv='dv_random_forest.bin'):
        if os.path.dirname(filename_model):
            os.makedirs(os.path.dirname(filename_model), exist_ok=True)
        if os.path.dirname(filename_dv):
            os.makedirs(os.path.dirname(filename_dv), exist_ok=True)
        with open(filename_model, 'wb') as f:
            pickle.dump(self.model, f)
        with open(filename_dv, 'wb') as f:
            pickle.dump(self.dv, f)

    def load_model(self, filename:str='model_random_forest.bin'):
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)


if __name__ == "__main__":
    preproc_normalizers = json.load(open('data_normalizers.json', 'r'))
    random_forest = TrainRandomForest(
        hparams={
            'max_depth': 6, 
            'n_estimators': 40, 
            'max_features': 'log2', 
            'min_samples_split': 7, 
            'min_samples_leaf': 1, 
            'bootstrap': False},
        data_normalizers=preproc_normalizers,
        seed=42,
        path_train='dataset/split_train.csv',
        path_val='dataset/split_val.csv',
        path_test='dataset/split_test.csv',
    )

    path_model = 'models/model_random_forest.bin'
    path_dv = 'models/dv_random_forest.bin'
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Training random_forest with hparams on full training dataset:\n{random_forest.hparams}")
    random_forest.train()
    y_pred, y_pred_proba = random_forest.predict(random_forest.X_test)
    logging.info(f"Evaluation random_forest on test dataset:\n{print_metrics(random_forest.y_test, y_pred)}")
    logging.info(f"Saving model to '{path_model}', saving dv to '{path_dv}'")
    random_forest.save_model(path_model, path_dv)

    logging.info(f"Loading model from {path_model}")
    random_forest.load_model(path_model)
    y_pred, y_pred_proba = random_forest.predict(random_forest.X_test)
    logging.info(f"Evaluation random_forest (from bin file) on test dataset:\n{print_metrics(random_forest.y_test, y_pred)}")

    
