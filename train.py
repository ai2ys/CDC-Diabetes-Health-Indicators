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

class DatasetLoader(object):
    def __init__(
        self, 
        path_train:str,  
        path_test:str,
        path_val:str=None, 
        seed:int=None,
        shuffle:bool=False):
        self.df_train_full = pd.read_csv(path_train, index_col='ID')
        self.df_test = pd.read_csv(path_test, index_col='ID')

        if path_val is not None:
            self.path_val = pd.read_csv(path_val, index_col='ID')
            self.df_train_full = pd.concat([self.df_train_full, self.path_val])

        # randomize the order of the training data
        if shuffle is not None:
            self.train_full = self.df_train_full.sample(frac=1, random_state=seed)

        # scale/normalize data and create DictVectorizer object
        self.data_normalizers = self.create_data_normalizers()
        self.dv = DictVectorizer(sparse=False)
        self.dv.fit(self.preprocess_features(self.train_full).to_dict(orient='records'))

        self.X_train_full, self.y_train_full = self.create_X_y(self.df_train_full)
        self.X_test, self.y_test = self.create_X_y(self.df_test)


    def save_data_normalizers(self,  path_file:str):
        if os.path.dirname(path_file):
            os.makedirs(os.path.dirname(path_file), exist_ok=True)
        with open(path_file, 'w') as f:
            json.dump(self.data_normalizers, f, indent=4)
            f.close()

    def load_data_normalizers(self, path_file)-> dict:
        with open(path_file, 'r') as f:
            self.data_normalizers = json.load(f)
            f.close()
        return self.data_normalizers

    def save_dv(self, path_file:str):
        if os.path.dirname(path_file):
            os.makedirs(os.path.dirname(path_file), exist_ok=True)
        with open(path_file, 'wb') as f:
            pickle.dump(self.dv, f)
            f.close()
    
    def load_dv(self, path_file)-> dict:
        with open(path_file, 'rb') as f:
            self.dv = pickle.load(f)
            f.close()
        return self.dv

    def create_data_normalizers(self) -> (dict):
        df = self.df_train_full
        data_normalizers = {
            'BMI':{
                'mean': df.BMI.mean(),
                'std': df.BMI.std(),
            },
            'Age':{
                'mean': df.Age.mean(),
                'std': df.Age.std(),
            },
            'MentHlth':
            {
                'min': df.MentHlth.min().astype(float),
                'max': df.MentHlth.max().astype(float),
            },
            'PhysHlth':
            {
                'min': df.PhysHlth.min().astype(float),
                'max': df.PhysHlth.max().astype(float),
            },
            'GenHlth':
            {
                'min': df.GenHlth.min().astype(float),
                'max': df.GenHlth.max().astype(float),
            },
            'Education':
            {
                'min': df.Education.min().astype(float),
                'max': df.Education.max().astype(float),
            },
            'Income':
            {
                'min': df.Income.min().astype(float),
                'max': df.Income.max().astype(float),
            },
        }
        return data_normalizers

    def preprocess_features(self, df):
        preproc_normalizers = self.data_normalizers
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

    def create_X_y(self, df:pd.DataFrame) -> (pd.DataFrame, pd.Series):
        """Create X and y from the dataframe including extracting the target column and preprocessing the features and applying the DictVectorizer.

        Args:
            df (pd.DataFrame): The dataframe to be preprocessed including the target column

        Returns:
            (pd.DataFrame, pd.Series): The preprocessed X data and y data
        """
        target_column = 'Diabetes_binary'
        # define the feature columns
        feature_columns = [col for col in df.columns if col != target_column]

        # split the data into train and val data
        X = self.preprocess_features(df[feature_columns])
        X = self.dv.transform(X.to_dict(orient='records'))
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
            seed:int, 
            dataset_loader:DatasetLoader=None
            ):
        """Create a TrainRandomForest model.

        Args:
            hparams (dict): The hyperparameters of the model
            seed (int): The seed
            dataset_loader (DatasetLoader): The dataset loader
        """
        self.hparams = hparams
        self.seed = seed
        self.dataset_loader = dataset_loader
        self.model = RandomForestClassifier(
            random_state=self.seed, **self.hparams, class_weight='balanced', n_jobs=-1)

    def train(self):
        self.model.fit(
            self.dataset_loader.X_train_full,
            self.dataset_loader.y_train_full)

    def predict(self, X):
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)
        return y_pred, y_pred_proba

    def evaluate_on_test_split(self):
        y_pred, y_pred_proba = self.predict(self.dataset_loader.X_test)
        metrics_str = print_metrics(self.dataset_loader.y_test, y_pred)
        return metrics_str

    def save_model(
        self, 
        filename_model:str='model_random_forest.bin', 
        filename_dv='dv.bin',
        filename_data_normalizers:str='data_normalizers.json',
        ):
        if os.path.dirname(filename_model):
            os.makedirs(os.path.dirname(filename_model), exist_ok=True)
        with open(filename_model, 'wb') as f:
            pickle.dump(self.model, f)
        self.dataset_loader.save_dv(filename_dv)
        self.dataset_loader.save_data_normalizers(filename_data_normalizers)

    def load_model(
        self, 
        filename:str='model_random_forest.bin', 
        filename_dv:str='dv.bin',
        filename_data_normalizers:str='data_normalizers.json',
        ):
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)
        self.dataset_loader.load_dv(filename_dv)
        self.dataset_loader.load_data_normalizers(filename_data_normalizers)


if __name__ == "__main__":
    seed = 42
    path_model = 'models/model_random_forest.bin'
    path_dv = 'models/dv.bin'
    path_data_normalizers = 'models/data_normalizers.json'

    dataset_loader = DatasetLoader(
        path_train='dataset/split_train.csv', 
        path_val='dataset/split_val.csv',
        path_test='dataset/split_test.csv',
        seed=seed,
        shuffle=True)
    random_forest = TrainRandomForest(
        hparams={
            'max_depth': 6, 
            'n_estimators': 40, 
            'max_features': 'log2', 
            'min_samples_split': 7, 
            'min_samples_leaf': 1, 
            'bootstrap': False},
        seed=seed,
        dataset_loader=dataset_loader
    )

    logging.basicConfig(level=logging.INFO)
    logging.info(f"Training random_forest with hparams on full training dataset:\n{random_forest.hparams}")
    random_forest.train()
    metrics_str = random_forest.evaluate_on_test_split()
    logging.info(f"Evaluation random_forest on test dataset:\n{metrics_str}")
    logging.info(f"Saving model to '{path_model}', saving dv to '{path_dv}', saving data normalizers to '{path_data_normalizers}")
    random_forest.save_model(path_model, path_dv, path_data_normalizers)

    logging.info(f"Loading model from '{path_model}', loading dv '{path_dv}', loading data normalizers '{path_data_normalizers}")
    random_forest.load_model(path_model, path_dv, path_data_normalizers)

    metrics_str = random_forest.evaluate_on_test_split()
    logging.info(f"Evaluation random_forest (from bin file) on test dataset:\n{metrics_str}")
