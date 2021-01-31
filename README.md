
# Heart Failure Predication

## Table of Contents
* ### Project Summary
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
* Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure. Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.  
* In this project, I aim to develop 2 ML models, the first using Hyperdrive to tune the hyperparameters of a Logisitic Regression model and the second using Azure's AutoML to find the model that best fits our data. The two models will be compared together and the best model will be deployed via Azure as a webservice. 


## Dataset  
Our dataset consists of 299 training examples and 13 features, we aim to predict the feature **DEATH_EVENT** which may have the value 1 in case of death due to heart faulure and 0 in case of survival.

### Overview
This dataset is available via Kaggle. The dataset includes 13 features that all contribute to the factors that cause Cardiovascular diseases such as age, anaemia and diabetes.  
Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure. Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

### Task
This is a binary classification task where our target column is **DEATH_EVENT** which may have the value 1 in case of death due to heart faulure and 0 in case of survival.

### Access
The data has been downloaded from [Kaggle](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) then uploaded to Azure Machine Learning Studio's Datasets where it'll be used to train the models.

## Automated ML
For the AutoML pipeline the following steps were done:
* Set up our environment and workspace. 
* Creating a Compute Cluster or using an existing one.
* Importing Our dataset.
* Configuring the AutoML run then submitting it. 
* Saving the best run & registring the model.
* Deploying the best model to a webservice. 
* Sending a request to the deployed webservice.


The AutoML configuration included the following:  
  * Setting the experiment_timeout_minutes to 30 minutes.
  * Defining task which is a *classification task*.
  * Setting primary_metric which is *Accuracy*.
  * Choosing a compute target.  
  * Defining the training data and target column, which is **DEATH_EVENT**.
  * Setting the number of cross validations which is in this model equals 3.
  * Enable auto-featurization to allow feature engineering, scaling and normalizing the data.
  * Enabling early_stopping to save computational power in case the model runs for too long.


### Results
* The best model was the **Voting Ensemble** model with an accuracy of **0.85956**.  
Best Run Id:  
![Best Run ID](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/id.png)
* Voting Ensemble is an ensemble machine learning model that combines the predictions from multiple other models. It is a technique that may be used to improve model performance, ideally achieving better performance than any single model used in the ensemble, that involves summing the predictions made by classification models.
* This result could be improved by increasing the experiment timeout time to allow automl to explore more models, which might be increase the accuracy. 

#### **AutoML Run Widget:**
![Run Details](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/run%20details%201.PNG)
* **AutoML runs accuracy:**  
![Run Details](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/run%20details%207.PNG)
**AutoML evaluated models:**  
![Run Details](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/run%20details%204.PNG)
![Run Details](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/run%20details%205.PNG)
![Run Details](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/run%20details%206.PNG)

**AutoML Best Run Metrics with the best run ID:**
![Best Run](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/best%20run%20metrics.PNG)
![Best Run](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/best%20run%20metrics2.PNG)

**Best Run from *Experiments* tab:**
![Best Run](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/best%20run%20automl2.PNG)

**Best Run Metrics from *Experiments* tab:**
![Best Metrics](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/best%20run%20metrics3.PNG)


## Hyperparameter Tuning

* We first need to prepare our train.py script following these steps:  
  * Importing the csv file containing the marketing campaigns data into our dataset.
  * Cleaning the dataset, which included droping NaN values.  
  * Splitting our dataset into training set (80% of the data) & test set (20% of the data.)   
  * Creating a Logistic Regression model using sci-kit learn.  
  * Creating a directory to save the generated model into it.  
  * After the train.py script is ready, we choose a proper parameter sampling method for the inverse regularization paramter(C) & the maximum number of iterations(max_iter),    early termination policy and an estimator to create the HyperDriveConfig.  

* The HyperDriveConfig was configured using the following:  
                             the estimator we created for the train.py,  
                             Paramater sampling method chosen,  
                             The early termination policy chosen,  
                             primary_metric_name, which is the Accuracy,  
                             primary_metric_goal, which is to maximize the primary metric,  
                             max_total_runs=4,  
                             max_concurrent_runs=4  
  * Then we submit the hyperdrive run.  
  * Once the run is complete, we choose the best run (the run that achieved the maximum accuracy), register the model generated and save it.
 
**The following diagram summarizes the workflow:**  
![Scikit-learn Pipeline](https://github.com/dinaabdulrasoul/optimizing-an-ml-pipeline/blob/master/hyperdrive_pipeline.PNG)  

**Algorithm**   
Logistic Regression is a supervisied binary classification algorithm that predicts the probability of a target varaible, returning either 1 or 0 (yes or no).  

**Parameter Sampler**  
* Here we use RandomParamaterSampler to determine the best values of the hyperparameters: **inverse regularization parameter, C** and **maximum number of iterations, max_iter**. 
* In this sampling algorithm, parameter values are randomly chosen from a set of discrete values or a distribution over a continuous range. Random Sampling is a great sampler to avoid bias, usually achieves great performance and it helps in discovering new hyperparameter values.
* Regularization strength is sampled over a uniform distribution with a minimum value of 0.5 and max value of 1, while the maximum number of iteration is sampled from a dicrete set of values which are 16, 32, 64 or 128.

**Early Stopping Policy**  
For this pipeline, Bandit Policy has been used, which is an early termination policy based on slack criteria, and the evaluation interval.
* Slack_factor is the ratio used to calculate the allowed distance from the best performing experiment run.  
* Evaluation_interval is the frequency for applying the policy.    
The benefits of this stopping policy* is that any run that doesn't fall within the slack factor will be terminated so this helps us in making sure the experiment doesn't run for too long and burn up a lot of resources while trying to find the optimal paramater value. 


### Results
 The best value of the Accuracy was found to be: **0.81667** using the following hyperparemeter values:
 ![hyperdrive](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/hp%20details.PNG)

#### **Hyperdrive Run Details Widget:**
![hyperdrive](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/hp3.PNG)
* **Hyperdrive Accuracy:** 
![hyperdrive](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/hp5.PNG)
* **2D Scatter Chart:**  
![hyperdrive](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/hp6.PNG)
* **3D Scatter Chart:** 
![hyperdrive](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/hp7.PNG)

**Hyperdrive Logs:**  
![hyperdrive](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/hp%20logs.PNGG)

**Hyperdrive Run from *Experiment* tab:**  
![hyperdrive](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/hp1.PNG)
![hyperdrive](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/hp2.PNG)

**Note:** The results achieved here could be improved by increasing the number of total runs from 4 to a higher number like 50, but that would need more computational power.


## Model Deployment

**Registered Models from *Models* tab:**  
![Models](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/models.PNG)
**Note:** *model* is the hyperdrive model while *automl_model* is the AutoML model.

As we can see, the **AutoML Voting Ensemble** model performed better than the logistic regression model tuned with **Hyperdrive** in terms of *accuracy*. 
The following steps have been performed to deploy the model and interact with it:  
* Saving the best run model.
* Writing the scoring script.
* Creating an ACI deployment configuration with key-based authentication enabled.
* Configuring *InferenceConfig* using the environment & the scoring script created as an entry script.
* Deploying the model to an ACI webservice. 
**The deployed model from *Endpoints* tab:**  
![Endpoint of AutoML run](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/deploy1.PNG)  
![Endpoint of AutoML run](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/deploy1.PNG) 

* A request is sent to the webservice endpoint.
* The following image shows the post request sent to the request API as a JSON document as well as the output of that request:
![Rest call to endpoint run](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/json.PNG) 

To send a request to the endpoint the following steps are followed:
* Getting the primary key, JSON data and scoring URI. 
* The JSON data should follow the following structure:
```
    data= { "data":
        [
            {
                'age': 60,
                'anaemia': 245,
                'creatinine_phosphokinase': 0,
                'diabetes': 0,
                'ejection_fraction': 38,
                'high_blood_pressure': 1,
                'platelets': 163000,
                'serum_creatinine': 50,
                'serum_sodium':100,
                'sex':1,
                'smoking':1,
                'time':7
                
                
            }
        ]
        }
```
        
* Creating the header with key *Content-Type* and value *application/json*.
* Setting the value of Authorization with Bearer token and primary key.
* Finally we send the Post-request to the web-service. 

## Screen Recording
[Screencast](https://youtu.be/Li8YZ5ysXY4)

## Standout Suggestions
Application Insights have been enabled for the deployed model. 
![Enabled](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/deploy2.PNG)

**The applications's insights:**  
![App Insights](https://github.com/dinaabdulrasoul/Heart-Failure-Prediction/blob/main/screenshots/more_screenshots/app%20insights.PNG) 

## Future Improvements
* Some of the imporvements might be using a different sampling paramater when configuring hyperdrive, for example: BayesianSampling or GridSampling, they might take up more time and resources however that might improve the accuracy.
* Also we can try using a different termination policy or not using a termination policy at all, for example with BayesianSampling method as the data we have is not that large so an early termination policy is not really necessary.
* For the AutoML configuration, we can try increase the timeout to 1 hour instead of 30 minutes. 
