
# CDC Diabetes Health Indicators<br>*(MLZoomCamp midterm project)*

This project uses the [CDC Diabetes Health Indicators dataset](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) dataset for training a model to predict if a patient is diabetic/pre-diabetic or does not have diabetes based on the patients heath records.


**Problem:** Diabetes is a severe illness that can lead to serious health problems such as heart disease, blindness, kidney failure, and so on. Detecting the illness in an early stage can help to prevent or delay these health problems.

**Task:** This midterm project aims to build a service that predicts whether a patient has diabetes, is pre-diabetic, or healthy using data provided by the "Diabetes Health Indicators Dataset" provided by the CDC. 

More information about the dataset can be found in the [Dataset Information](#📋-dataset-information) section.

The dataset used in this project was created to to better understand the relationship between lifestyle and diabetes in the US and the creation was funded by the [CDC (Center for Disease Control and Prevention)](https://www.cdc.gov/).

## 📋 Dataset Information

🔗 Dataset page: [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

The [CDC Diabetes Health Indicators dataset](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) is available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/). 

- Each row of the dataset represents a person participating in the study.
- The dataset contains 21 feature variables (categorical and integer) and 1 target variable (binary).
- Cross validation or a fixed train-test split could be used for data splits. 
- It contains sensitive data such as gender, income, and education level. 
- Data preprocessing was performed by bucketing of age. The dataset has no missing values. 


**Quoted from the dataset page**<br>

> *"The Diabetes Health Indicators Dataset contains healthcare statistics and lifestyle survey information about people in general along with their diagnosis of diabetes. The 35 features consist of some demographics, lab test results, and answers to survey questions for each patient. The target variable for classification is whether a patient has diabetes, is pre-diabetic, or healthy."*

**Remark on the quote above**<br>
The quote states that the dataset contains 35 features. However, the dataset page further states

>|  | Information |
>| :--- | :--- |
>| Dataset Characteristics | Tabular, Multivariate |
>| Subject Area | Life Science |
>| Associated Tasks | Classification |
>| Feature Type | Categorical, Integer |
>| \# Instances | 253680 |
>| \# Features | 21 |

💡 We will check this discrepancy in when digging into the dataset during the EDA (Exploratory Data Analysis).

➡️ From the dataset information we can see that the task will be a binary classification task with 21 features and 1 target variable.


**Download is provided via**

1. Python API using the `ucimlrepo` package.
    - When using the `ucimlrepo` package for downloading the data, there is an additional download link provided in its `metadata.data_url` from which the dataset CSV file can be accessed directly.
    https://archive.ics.uci.edu/static/public/891/data.csv
1. On the project page (https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) there is a reference to the dataset source which redirects to Kaggle:<br>
https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset


💡 In this project, we utilize the ucimlrepo Python package to download the initial dataset. To ensure reproducibility, all relevant data for exploratory data analysis (EDA) and training will be stored locally. This approach safeguards against potential issues, such as unavailability or changes to the dataset in the UCI Machine Learning Repository over time.

📥 How the dataset was downloaded and stored locally is described in the EDA notebook.


## 🛠️ Setup Virtual Environment for EDA and Training

Setting up the virtual environment using **Miniconda** with Python `3.10.12`.
1. Installing Miniconda
    - Depending on your Host OS following instruction from:\
    https://docs.conda.io/projects/miniconda/en/latest/

1. Creating the virual environment using Python 3.10.12
    ```bash
    conda create --name mlzoomcamp-midterm python=3.10.12
    ```	
1. Activating the virtual environment
    ```bash
    conda activate mlzoomcamp-midterm
    ```
    - Install the requirements within the activated virtual environment `(mlzoomcamp-midterm)`
        ```bash
        pip install -r requirements-eda.txt
        ```
    - Start JupyterLab
        ```bash
        jupyter lab
        ```

## 📊 EDA (Exploratory Data Analysis) and 🧠 Model Training

The Jupyter Notebook [`eda.ipynb`](eda.ipynb) contains

1. EDA for the dataset
1. Training and evaluating different models
    - Different algorithms (tree-based and linear models)
    - Different hyperparameters

🐍 Activate the virtual environment and start JupyterLab as described in section [🛠️ Setup Virtual Environment for EDA & Training](#🛠️-setup-virtual-environment-for-eda-and-training)

```bash
# activate virtual environment
conda activate mlzoomcamp-midterm
# within the activated environment start JupyterLab
jupyter lab
```


## Export notebook to Python script

<!--  -->

## Model Deployment

<!-- 1. Within the activated environment `(mlzoomcamp-midterm)...$`
    - Install `pipenv` 
        ```bash
        pip install pipenv==2023.10.24
        ```

    1. Create a `Pipfile` and `Pipfile.lock` based on the provided `requirements-eda.txt`
        ```bash
        pipenv install -r requirements-eda.txt
        ```
    1. Activate the `pipenv shell` for this project
        ```bash
        pipenv shell
        ``` -->


<!-- Deploying model using e.g. Flask -->

## Containerization

## Cloud Deployment













---

## MLZoomCamp Midterm Project General Information
> General information about he MLZoomCamp Midterm Project can be found here:
https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/projects#midterm-project

>Information for cohort 2023 can be found here:
https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/cohorts/2023/projects.md#midterm-project