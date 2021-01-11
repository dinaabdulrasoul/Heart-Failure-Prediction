
# Heart Failure Predication

## Table of Contents
* ### Project Summary
* ### Project Set Up and Installation
* ### Dataset
* ### Overview
* ### Task
* ### Access
* ### Automated ML
  * Results
* ### Hyperparameter Tuning
  * Results
* ### Model Deployment
* ### Screen Recording
* ## Standout Suggestions
## Project Summary  
Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure. Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.  
In this project, I aim to develop 2 ML models, using Hyperdrive to tune the hyperparameters of a Logisitic Regression model and Azure's AutoML to find the model that best fits our data. The two models will be compared together and the best model will be deployed via Azure, consumed and published through an endpoint. 

## Project Set Up and Installation  
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset  
Our dataset consists of 299 training examples and 13 features, we aim to predict the feature **DEATH_EVENT** which may have the value 1 in case of death due to heart faulure and 0 in case of survival.

### Overview
*TODO*: Explain about the data you are using and where you got it from.

### Task
This is a binary classification task where our target column is **DEATH_EVENT** which may have the value 1 in case of death due to heart faulure and 0 in case of survival.

### Access
The data has been downloaded from [Kaggle](Kaggle https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) then uploaded to my github where it's later used as a dataframe in the notebook provided. 

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
