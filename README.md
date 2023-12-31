# CDC Diabetes Health Indicators<br>*(MLZoomCamp midterm project)*

Table of Contents: 
1. [📖 Introduction](#introduction)
1. [📋 Dataset Information](#dataset-information)
1. [📊 EDA and 🧠 Model Training](#eda-and-model-training)
    1. [🛠️ Virtual Environment Setup](#virtual-environment-setup)
    1. [📓 Run Jupyter Notebook](#run-jupyter-notebook)
    1. [💡Information from EDA and Model Training](#information-from-eda-and-model-training)
    1. [📤 Export Notebook to  Python Script](#export-notebook-to-python-script)
1. [🧩 Model Deployment](#model-deployment)
    1. [⚙️ Test Model Deployment](#test-model-deployment)
1. [🐋 Containerization](#containerization)
    1. [🛠️ Create Pipfile and Pipfile.lock](#create-pipfile-and-pipfilelock)
    1. [▶️ Run Docker Container and Test the Service](#run-docker-container-and-test-the-service)
1. [☁️ Cloud Deployment](#cloud-deployment)
    1. [📋 Prerequisites](#prerequisites)
    1. [▶️ Run and Test Cloud Deployment](#run-and-test-cloud-deployment)

1. [MLZoomCamp Midterm Project General Information](#mlzoomcamp-midterm-project-general-information)

## Introduction

📖
This project uses the [CDC Diabetes Health Indicators dataset](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) that can be used for training a model to predict if persons are diabetic/pre-diabetic or non-diabetic diabetes based on their heath records.
The dataset was created to to better understand the relationship between lifestyle and diabetes in the US and the creation was funded by the [CDC (Center for Disease Control and Prevention)](https://www.cdc.gov/).

**Problem:** Diabetes is a severe illness that can lead to serious health problems such as heart disease, blindness, kidney failure, and so on. Detecting the illness in an early stage can help to prevent or delay these health issues.

**Task:** This midterm project aims to build a service that predicts whether a patient is (pre-)diabetic or healthy using the previous mentioned data provided by the "CDC Diabetes Health Indicators Dataset". 

More information and insights about the dataset can be found in the [Dataset Information](#dataset-information) section.

## Dataset Information
📋 Information regarding the dataset
- Dataset source and dataset download location
- Features variables
- Target variable

🔗 Dataset page:
- [CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- [DOI 10.24432/C53919](https://doi.org/10.24432/C53919)

The [CDC Diabetes Health Indicators dataset](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) is available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/). 

Information from the dataset page:
- Each row of the dataset represents a person participating in the study.
- The dataset contains 21 feature variables (categorical and integer) and 1 target variable (binary).
- Cross validation or a fixed train-test split could be used for data splits. 
- The dataset contains sensitive data such as gender, income, and education level. 
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


💡 In this project, we utilize the `ucimlrepo` Python package to download the initial dataset. To ensure reproducibility, all relevant data for exploratory data analysis (EDA) and training will be stored locally in the ([`./dataset`](dataset) folder. This approach safeguards against potential issues, such as unavailability or changes to the dataset in the UCI Machine Learning Repository over time.

📥 How the dataset was downloaded and stored locally is described in the EDA notebook [`notebook.ipynb`](notebook.ipynb). The dataset and parts of the metadata are downloaded the [`notebook.ipynb`](notebook.ipynb) and stored in the [`./dataset`](dataset) folder locally.
- dataframe - [`./dataset/data.csv`](dataset/data.csv)
- information about variables - [`./dataset/variables.csv`](dataset/variables.csv)
- metadata (only some parts of it) - [`./dataset/metadata_partially.json`](dataset/metadata_partially.json)

📊 During EDA the generated dataset splits (train, validation, test) have been downloaded to separate files. These will be used later on for running the training on the 'full training' dataset (train + validation) in the final [`train.py`](train.py) script. 
- [`split_train.csv`](`split_train.csv`)
- [`split_val.csv`](`split_val.csv`)
- [`split_test.csv`](`split_test.csv`)


## EDA and Model Training

📊 EDA stands for Exploratory Data Analysis, which is the process of analyzing data sets to summarize the main characteristics, often including visualizations of the data. EDA is used for seeing what the data can tell us about the data.

🧠 Model training is the process of training a machine learning model to make predictions based on data. The model training process includes the following steps:

1. Data preprocessing
1. Model training
1. Model evaluation

📓 The EDA and model training will be performed in a Jupyter Notebook [`notebook.ipynb`](notebook.ipynb). In this notebook, we will perform the following steps:

1. Exploratory Data Analysis (EDA)
1. Data preprocessing
1. Model training
    - Different algorithms (tree-based and linear models)
    - Different hyperparameters
1. Model evaluation
1. Model selection

All required steps for setting up the virtual environment to run the notebook are described in the [🛠️ Virtual Environment Setup](#virtual-environment-setup) section.

🐍 For training the final model after the model selection step we will extract the required Python code to a script called [`train.py`](train.py). This notebook will be used to train the final model and save it to disk. Steps covered in the [`train.py`](train.py) script will be:

1. Data preprocessing
1. Model training
1. Model evaluation
1. Model storage

### Virtual Environment Setup

🐍 Setting up the virtual environment for 📊 EDA and 🧠 Model Training using **Miniconda** with Python `3.10.12`. All required packages will be installed from the [`requirements-eda.txt`](requirements-eda.txt) file. There the packages are listed with their version number to ensure reproducibility.

1. Installing Miniconda
    - Follow the instruction from for your host OS:\
    https://docs.conda.io/projects/miniconda/en/latest/

1. Creating the virtual environment using Python 3.10.12
    ```bash
    conda create --name mlzoomcamp-midterm python=3.10.12
    ```	

1. Activating the virtual environment
    ```bash
    conda activate mlzoomcamp-midterm
    ```
    - The command prompt should now indicate that the virtual environment is activated and show the name of the virtual environment in parentheses `(mlzoomcamp-midterm)`.\
    Within the activated virtual environment `(mlzoomcamp-midterm)` perform the following steps:

        1. Install the requirements from the [`requirements-eda.txt`](requirements-eda.txt) 
            ```bash
            pip install -r requirements-eda.txt
            ```
        1. Start JupyterLab to check if the installation was successful
            ```bash
            jupyter lab
            ```


> *Additional information:*     
>The commands above worked in WSL2 (Windows Subsystem for Linux) on Windows 11 and should be the same on Linux. The `conda` version installed on my system is `23.0.9` ([Conda command reference `23.9.x`](https://docs.conda.io/projects/conda/en/23.9.x/commands/index.html).\
> In case you are using a different `conda` version and the `conda` commands do not work on your system, check the `conda` cheat-sheet of your installed `conda` version for the correct commands.


### Run Jupyter Notebook

Information on ▶️ running the 📓 Jupyter Notebook [`notebook.ipynb`](notebook.ipynb).

The previous created virtual environment `(mlzoomcamp-midterm)` has JupyterLab installed. In order to start JupyterLab, the virtual environment needs to be activated first. Activate the virtual environment that we created in the previous section [🛠️ Virtual Environment Setup](#virtual-environment-setup).

```bash
# navigate to project directory, the location and command might differ on your system
cd CDC-Diabetes-Health-Indicators

# activate the virtual environment using 'mlzoomcamp-midterm
conda activate mlzoomcamp-midterm

# within the activated environment indicated by '(mlzoomcamp-midterm)' start JupyterLab 
jupyter lab
```

### Information from EDA and Model Training

💡Insights and Results from the 📊 EDA (exploratory data analysis) and the 🧠 Model Training.


#### EDA - Variables (Target and Features) 

💡Information revealed during the 📊 EDA from the dataset and its metadata.

The following data was retrieved after downloading the dataset in the EDA [`notebook.ipynb`](notebook.ipynb) using the `ucimlrepo` Python package and storing the 'variables' information to [dataset/variables.csv](dataset/variables.csv). In order to not duplicate data in multiple files the following data hast been stored here and not in the EDA notebook.

|  | Type | Description | 
| --- | --- | --- |
| ID | Integer | Patient ID |



| Target variable | Type | Description | 
| --- | --- | --- |
| Diabetes_binary | Binary | 0 = no diabetes<br>1 = prediabetes or diabetes |

Features in are sorted in the table below using their data type:
- Integer
- Categrical
- Binary

> Information for binary features (except for feature `Sex`):
> - `0` = `no` 
> - `1` = `yes`

| Features | Type | Description | 
| --- | --- | --- |
| BMI | Integer | Body Mass Index|
| MentHlth | Integer | Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good?<br> scale 1-30 days |
| PhysHlth | Integer | Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good?<br> scale 1-30 days |
|  |  |  |
| GenHlth | Integer (Categorical) | Would you say that in general your health is: scale 1-5<br>1 = excellent<br>2 = very good<br> 3 = good<br> 4 = fair<br> 5 = poor |
| Age | Integer (Categorical) | Age,13-level age category (_AGEG5YR see codebook)<br>1 = 18-24<br>9 = 60-64<br> 13 = 80 or older |
| Education | Integer (Categorical) | Education level (EDUCA see codebook) scale 1-6<br>1 = Never attended school or only kindergarten<br>2 = Grades 1 through 8 (Elementary)<br>3 = Grades 9 through 11 (Some high school)<br>4 = Grade 12 or GED (High school graduate)<br>5 = College 1 year to 3 years (Some college or technical school)<br>6 = College 4 years or more (College graduate) |
| Income | Integer (Categorical) | Income scale (INCOME2 see codebook) scale 1-8<br> 1 = less than $10,000<br> 5 = less than $35,000<br> 8 = $75,000 or more" |
|  |  |  |
| Sex | Binary | Sex, 0 = female 1 = male |
| HighBP | Binary | High blood preasure |
| HighChol | Binary | High cholesterol |
| CholCheck | Binary | Cholesterol check in 5 years |
| Smoker | Binary | Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes] |
| Stroke | Binary | (Ever told) you had a stroke. |
| HeartDiseaseorAttack | Binary | Coronary heart disease (CHD) or myocardial infarction (MI) |
| PhysActivity | Binary | Physical activity in past 30 days - not including job< |
| Fruits | Binary | Consume Fruit 1 or more times per day |
| Veggies | Binary | Consume Vegetables 1 or more times per day |
| HvyAlcoholConsump | Binary | Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)|
| AnyHealthcare | Binary | "Have any kind of health care coverage, including health insurance, prepaid plans such as HMO, etc. |
| NoDocbcCost | Binary | Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? |
| DiffWalk | Binary | Do you have serious difficulty walking or climbing stairs? |


#### EDA - Missing Values, Duplicates, Imbalances, etc.

💡Information revealed about the dataset during 📊EDA regarding

- Missing values
    - ✅ As stated in the dataset information, the dataset has no missing values
- Duplicates
    - ✅ There are duplicate rows in the dataset when not taking into account the patient ID. This is due to the fact that feature variables are categorical, binary and integer. The integer values have either only value ranges between 1 and 30 or are discrete values although the original values were floating point (feature `BMI`). Therefore these rows represent different patients that just represent the same feature variables due to their nature.
- Imbalances
    - ✅ The dataset is highly imbalanced with respect to the target varibale
        - 14% (pre-)diabetic
        - 86% non-diabetic


The [BMI (body mass index)](https://en.wikipedia.org/wiki/Body_mass_index) is calculated using the following formula. The result is a floating point number, but in the dataset the BMI is stored as an integer. This means that the BMI is rounded to the next integer. 

$$
BMI_{\text{float}} = \frac{mass_{kg}}{height_{m}^2}
$$

$$
BMI_{\text{integer}} = \text{integer}\left(\frac{mass_{kg}}{height_{m}^2} + 0.5 \right)
$$




### Export Notebook to Python Script
📤 The code for training the final model got exported to the 🐍 Python script [`train.py`](train.py). The script will covers the following tasks:

- Loading the dataset splits: train, validation, and test
- Creating a test dataset consisting of the test split
- Creating a 'full training' dataset consisting of training split and validation split
- Training the model on the 'full training' (train + validation) dataset
- Evaluating the model on the test dataset
    - Printing the metrics to the command line
- Saving the following data to files (bin and json)
    - Model
    - DictVectorizer (fitted on 'full training' dataset)
    - Normalization values (determined on 'full training' dataset in order to normalize the value ranges of some feature variables) 

🐍 For running the [`train.py`](train.py) script make sure the development environment defined in section [🛠️ Virtual Environment Setup](#virtual-environment-setup) is activated before running the following commands.

```bash
# 🐍 Activate the development environment
conda activate mlzoomcamp-midterm

# ▶️ Execute the training script
python train.py
```

🎲 We will now randomly sample an entry from the test dataset for testing the model later on. For this purpose the script [`sample_from_test.py`](sample_from_test.py) is used. The script will randomly sample a test dataset entry (row) and store it as `JSON` file [`test_sample.json`](test_sample.json).

```bash
# 🐍 Activate the development environment, if not already done
conda activate mlzoomcamp-midterm

# 🎲 sample randomly without seed point
python sample_from_test.py

# 🎲 sample randomly using a specific seed point
python sample_from_test.py --seed 1234
```

This [`test_sample.json`](test_sample.json) will be used when testing the model during the next step the Model Deployment.

## Model Deployment

🧩 For deploying create a new virtual environment for testing the deployment.

1. Create the a environment using Python 3.10.12
    ```bash
    conda create --name deployment-midterm python=3.10.12
    ```	

1. Activate the virtual environment
    ```bash
    conda activate deployment-midterm
    ```
    
    Within the activated virtual environment `(deployment-midterm)` install the requirements from the [`requirements-deployment.txt`](deployment-eda.txt) 
    ```bash
    pip install -r requirements-deployment.txt
    ```

### Test Model Deployment

⚙️ Test the deployment script starting the predict service will require two terminal windows

1. Terminal windows #1: Run the predict service

    <!-- #waitress-serve --listen=0.0.0.0:9696 predict:app -->
    ```bash
    # activate 'deployment-midterm', if not already activated 
    conda activate deployment-midterm 
    # start the predict service
    python predict.py
    ```

1. Terminal window #2: Execute the http-request using [`test_sample.py`](test_sample.py), which will use the sample from [`test_sample.json`](test_sample.json)

    ```bash	
    # activate 'deployment-midterm', if not already activated 
    conda activate deployment-midterm
    # test the predict service using the sample from 'test_sample.json'
    python test_predict.py
    ```



## Containerization

🐋 Putting the prediction service in a Docker container, which requires Docker being installed on your system. 

### Create Pipfile and Pipfile.lock

🛠️ Create a Pipfile and Pipfile.lock for containerization using `pipenv`
1. Install `pipenv` 
    ```bash
    pip install pipenv==2023.10.24
    ```
1. Create a `Pipfile` and `Pipfile.lock` based on the provided `requirements-eda.txt`
    ```bash
    pipenv install -r requirements-deployment.txt
    ```
<!-- 1. Activate the `pipenv shell` for this project
    ```bash
    pipenv shell
    ``` -->


### Run Docker Container and Test the Service

The Docker image `docker pull ai2ys/mlzoomcamp-midterm-project:0.0.0` has been pushed to the 🐋 DockerHub registry. Therefore you can run the container without prior building by just running the container, which will pull the image from DockerHub.


<!-- 
Command used for pushing the Docker image
```bash
docker push ai2ys/mlzoomcamp-midterm-project:0.0.0
```

Command used for pulling the Docker image
```bash	
docker pull ai2ys/mlzoomcamp-midterm-project:0.0.0
``` -->

1. *Optional:* Building the Docker image `ai2ys/mlzoomcamp-midterm-project:0.0.0`
    ```bash
    docker build -t ai2ys/mlzoomcamp-midterm-project:0.0.0 .
    ```

1. Running the Docker container (terminal windows #1)
    ```bash	
    docker run --rm -p 9696:9696 ai2ys/mlzoomcamp-midterm-project:0.0.0
    ```

1. Testing the prediction service in the Docker container from the virtual environment `(deployment-midterm)`.\
Open a new terminal window and execute the following commands (terminal window #2)

    ```bash
    # activate the virtual environment
    conda activate deployment-midterm
    # run the test script
    python test_predict.py	
    ```


##  Cloud Deployment
☁️ Instructions for the cloud deployment using AWS Elastic Beanstalk.

### Prerequisites
- Amazon AWS Account<br>
    For this task an AWS account is required. Please create an AWS account following the instructions from [Machine Learning Bookcamp - Creating an AWS Account](https://mlbookcamp.com/article/aws).

- [Installing the EB CLI (elastic beanstalk command line interface)](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-install-advanced.html)<br>
    ```bash	
    # create virtual environment for AWS Elastic Beanstalk
    conda create --name awsebcli python=3.10.12 

    # install AWS Elastic Beanstalk CLI
    pip install awsebcli==3.20.10

    # activate the virtual environment
    conda activate awsebcli

    # check AWS Elastic Beanstalk CLI version
    eb --version
    ```

### Run and Test Cloud Deployment

🎞️ Video of cloud deployment showing all steps below: 🔗 [https://youtu.be/eu-TP17kvwc](https://youtu.be/eu-TP17kvwc)


▶️ Steps for creating and running the prediction service on AWS Elastic Beanstalk.

- Initialize AWS Elastic Beanstalk project
    ```bash	
    # activate the virtual environment
    conda activate awsebcli
    # initialize eb, select region, specify credentials
    eb init -p "Docker running on 64bit Amazon Linux 2023" -r eu-west-1 --profile <profile> mlzoomcamp-midterm-project
    ```
- Test locally using Elastic Beanstalk

    1. Terminal window #1: Using AWS Elastic Beanstalk to run the service locally
        ```bash
        # activate the virtual environment
        conda activate awsebcli
        eb local run --port 9696
        ```
    1. Terminal window #2: Run the [`test_predict.py`](test_predict.py) script
        ```bash
        # activate the virtual environment 'deployment-midterm'
        conda activate deployment-midterm
        python test_predict.py
        ```

- Test cloud deployment using Elastic Beanstalk
    1. Terminal window #1: Create the Elastic Beanstalk environment
        ```bash
        # activate the virtual environment 'awsebcli'
        conda activate awsebcli
        eb create mlzoomcamp-midterm-env
        ```
        When the service is running copy the URL to the clipboard 📋

    1. Terminal window #2: Run the [`test_predict.py`](test_predict.py) script
        In another terminal run the [`test_predict.py`](test_predict.py) and insert the URL 
        ```bash
        # activate the virtual environment 'deployment-midterm'
        conda activate deployment-midterm
        python test_predict.py --url <elastic beanstalk url>
        ```

- When we are done running the prediction service on AWS Elastic Beanstalk
    ```bash
    # if not already using activate the virtual environment
    conda activate awsebcli
    eb terminate mlzoomcamp-midterm-env
    ```
    


<!-- getting started https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/Gettineb
1. Sign in to your AWS account
1. Search for 'Amazon Elastic Beanstalk' 
    1. Select 'Create application' button

<!-- [Configure AWS Elastic Beanstalk CLI](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/eb-cli3-configuration.html)<br> -->






---

## MLZoomCamp Midterm Project General Information
> General information about he MLZoomCamp Midterm Project can be found here:
https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/projects#midterm-project

>Information for cohort 2023 can be found here:
https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/cohorts/2023/projects.md#midterm-project
