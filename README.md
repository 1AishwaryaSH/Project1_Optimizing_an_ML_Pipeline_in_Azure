# Optimizing an ML Pipeline in Azure
## (part of Udacity's Machine Learning Engineer with Microsoft Azure NanoDegree)

## Project Overview
This project aims at creating a machine learning model using the HyperDrive for optimizing the pipeline and comparing the results obtained with the model constructed using Azure AutoML.
To build the pipeline we use Python SDK and a Scikit-learn model.

## Project Summary
The best run model is the AutoML model "VotingEnsemble" with an Id: AutoML_10a12021-419a-46bc-9736-103ece2219b6_18 and Accuracy:0.9173899034676254.
It performs better than the logistic regression(Scikit-learn) model using HyperDrive optimized hyperparameters.

## Architecture
The project architecture involves the following major steps:
script file : train.py
* Import the dataset from the specified URL.
* Cleaning the Data.
* Splitting the data into train and test datasets with 8:2 ratio of total available data respectively.

jupyter notebook : udacity-project-Copy1.ipynb
* Creating a workspace for the pipeline.
* Using CPU clusters if already existing or create new cluster.
* I used Random sampling method for optimizing the hyperparameters 'C' and 'max_iter' using uniform and randint hyperdrive parameter expressions.
* The early termination policy is BanditPolicy with an evaluation interval of 2 and slack_factor of 10%.
* The hyperdrive is configured using the SKLearn estimator, hyperparameter sampler, and policy.
* Once the run is finished I registered the best model obtained using HyperDrive.
* For AutoML model
  * we obtain a csv webfile dataset.
  * configure by setting the parameters of AutoMLConfig.
  * submit the AutoML run.
  * Obtain the metrics from the model.
* Save the best model.


### About the DataSet
I used the data set containing a Bank's Marketing data that involves a target label of binary classification mentioning whether the customer will do a deposit or not, labelled with the name 'y'. This data set can be found at "https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv"

### Algorithm
We use the two class Binary classification algorithm to predict the discrete values {'yes' or 'no'} for the column labelled 'y' in the dataset.


### Hyperparameters
The two Hyperparameters used are the 'C' or inverse regularization parameter and 'max_iter' representing the maximum number of iterations that are allowed. To perform this I used RandomParameterSampler that allows to find the better values out of large space randomly with less computation power.

## Policy
BanditPolicy with an evaluation interval of 2 and slack factor of 10% is used to avoid the wastage of resources. Once the parameter values are reached it terminates the process.
![hyperdrive](az1.JPG) 

## AutoML
The AutoML model 'Voting Ensemble' gave an accuracy of 91.74%.
The model is configured with the parameters experiment timeout, compute target, task, primary metric,training data,label column name,iterations, max concurrent iterations, n cross validations. Here the task is classification and the iterations is fixed to 20 which specifies the number of times a model is iterated to find a better accuracy metric .The primary metrics can be accuracy, AUC,.... Cross validation is used as a technique to split original data into the train and test which I speified as 4 folds.

![iterations](autoML_iter.JPG)

## Comparision
In Hyperdrive I used logistic regression model for the prediction with hyperparameters namely C, max_iter, where as in AutoML multiple models were generated with less human intervention and less time which would otherwise needed more effort for tuning the hyperparameters and training these models.  
For this particular project on classification of customers in Bank Fixed Deposit the AutoMl model is more accurate.
The best model obtained using HyperDrive was found with an Accuracy: 0.9130500758725342 where as the best AutoMl model 'VotingEnsemble' had an Accuracy: 0.9173899034676254. So we conclude that use of AutoML.
Coming to the aspect of execution time I found that the hyperdrive took comparitively less time to generate the model.

## Areas of Improvement
I found that the dataset is imbalanced and further use of data cleaning techniques could help us get better accuracy models.

## Proof of Cleaning clusters
![delete](d1.JPG)


## References
https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-first-experiment-automated-ml
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-train
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-scikit-learn
https://azure.microsoft.com/en-in/resources/templates/101-machine-learning-compute-create-amlcompute/
https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/work-with-data/datasets-tutorial/train-with-datasets
https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/ml-frameworks/scikit-learn/train-hyperparameter-tune-deploy-with-sklearn/train-hyperparameter-tune-deploy-with-sklearn.ipynb
