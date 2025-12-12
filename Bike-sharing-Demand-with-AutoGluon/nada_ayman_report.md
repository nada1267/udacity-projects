# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### Nada Ayman Mostfa

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?

When i submit my predictions in the first submission(submission.csv )and second submission(submission_new_features.csv) it goes right but in the third submission i have some errors in the part of trainning many models using AutoGluon so i fix it and my third submission (submission_new_hpo)went right.
     
### What was the top ranked model that performed?
  WeightedEnsemble_L3 Model,with a RMSE -53.089702 

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
-First part:
I find exploratory analysis very important and useful and give easy overview for all features in the data and through histogram i discover the distrbution for each feauture.

-second Part:
i see that the datatime feauture most suitable to divide it for multiple feauters as year,month,day,hour.so in this way i can add additional features 


### How much better did your model preform after adding additional features and why do you think that is?
-First part:
The model's performance has clearly improved and it has been better than before.

-second Part:
I think because the model train on additional feauters and this can improve the performance especially these feauters has big effect to the result

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
I think here ass hyper parameters has't improve the model performance as i think the improvement because of adding new feautures as the second submission is the best one and after it the hyper parameters doen't improve the score

### If you were given more time with this dataset, where do you think you would spend more time?
-may be i will search if i can make it more bigger.

-study topics related the market demand i think it may be useful.


### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|timelimit|presets|hp-method|score|
|--|--|--|--|--|
|initial|time_limit = 600|presets='best_quality' |none	|1.80837|
|add_features|time_limit = 600 |presets='best_quality'|problem_type = 'regression'|0.61692|
|hpo|time_limit = 600|presets='best_quality'|tabular autogluon|0.68872|


### Create a line plot showing the top model score for the three (or more) training runs during the project.
 ![image](https://github.com/nada1267/myibmsupervisedclassification/assets/99268869/5e74a677-cc7b-40ee-921d-017a026672f1)


### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.
 
![image](https://github.com/nada1267/myibmsupervisedclassification/assets/99268869/a1ee6813-05d1-48d5-a5c4-9ec1901865db)

## Summary
AutoGluon  enable us to train many models so i can choose the best model for the problem and applying techniques as add new feutures can improve the performance of the model and set highparameters can effect on the performance but may be this effect not desirable so we should set it carfully.
