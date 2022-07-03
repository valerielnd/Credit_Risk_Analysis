# Credit_Risk_Analysis

## Project overview

Fast lending, a peer-to-peer lending services company, wants to use machine learning to predict credit risk. 
Loans present an opportunity and challenge for banks and other lending institutions. On the one hand, they create 
revenue with the interest they generate; on the other hand, there is a risk that borrowers won't repay, and 
the lending institution will lose money. As machine learning can process a large amount of data to arrive 
at a single decision, we will build and evaluate several machine learning models to assess credit risk and 
find an accurate way to identify suitable candidates for loans.


## Resources

To perform the analyses on this project, we used the credit data from LendingClub, a similar peer-to-peer lending services company:

[Project Dataset](https://help.lendingclub.com/hc/en-us/articles/215488038-What-do-the-different-Note-statuses-mean-)

To build and evaluate our machine learning models, we will employ the python library Scikit-learn, imbalanced-learn, and Pandas and Numpy.

## Project Deliverables

* Use Resampling Models to Predict Credit Risk
* Use the SMOTEENN Algorithm to Predict Credit Risk
* Use Ensemble Classifiers to Predict Credit Risk

## Analysis results

### Use Resampling Models to Predict Credit Risk

### Random oversampling

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. 
Therefore, we used imbalanced-learn and scikit-learn libraries to build and evaluate models with unbalanced classes.

Before we start building our models, the first step is to prepare the data.

We read the data file into a dataFrame:

![load_data](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/load_data.png)

The dataset includes the following columns:

![data_columns](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/data_columns.png)

So the dataFrame reveals one target: "loan_status," and all the other columns are features.
 
As the dataframe has columns where all values are null and null rows, we dropped them.

Since we are interested in loan status that is current or has not been paid yet, we 
removed from the dataFrame any loan data that have the status "Issued"

Then we convert the interest rate to a number and the target column values to low_risk and high_risk based on their values:

![final_data_prep](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/final_data_prep.png)

Machine learning algorithms typically only work with numerical data, categorical and text data must be converted to numerical data.

The following columns were encoded using the get_dummies() method wile defining the feature set:

![encoded_columns](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/encoded_columns.png)

![feature_set](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/feature_set.png)

We created our target set and check its balance:

![target_set](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/target_set.png)

![target_set_balance](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/target_set_balance.png)

As predicted, the existing classes in the dataset aren't equally represented.

Next, we split the data into training and testing sets. The models use the training dataset to learn from it. 
They will use the testing dataset to assess their performance. When splitting the data, we stratified it to divide it proportionally:

![split_data](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/split_data.png)

Since the imbalance between the two classes (high-risk and low-risk) can cause our models to be biased toward the majority class, 
we oversampled the minority class with the RandomOverSampler algorithm.

![resample_naive](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/resampled_naive.png)

As we can see, the minority class has been enlarged.

After preprocessing the features data, the logistic regression model, a popular classification model, is created and trained:

![regression_oversampling](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/regression_oversampling.png)

After fitting the model, we ran the following code to make predictions using the testing data:

![regression_over_predictions](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/regression_over_predictions.png)

We evaluated how well our model classifies loan applications by creating a confusion matrix: a table of true positives, false positives, true negatives, and false negatives.

![oversampling_cm](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/oversampling_cm.png)

The results show that:

* Out of 87 High-Risk applications, 52 were predicted to be High-Risk.
* Out of 87 High-Risk applications, 35 were predicted to be low-Risk.
* Out of 17118 low-Risk applications, 5952	 were predicted to be high-Risk.
* Out of 17118 low-Risk applications, 11166 were predicted to be low-Risk.

Next, we determined the model accuracy:

![oversampling_accuracy](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/oversamplig_accuracy.png)

The model achieved an accuracy score of 0.62. This value is not large enough to suspect overfitting.

Finally, we used the classification_report_imbalanced method to get the precision, sensitivity, and F1 score associated with the classes of the model:

![oversampling_report](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/oversampling_report.png)

The precision for high-risk application is 0.01, which indicate a large number of false positives, while it is perfect for low-risk application.

The sensitivity for high-risk applications is 0.60 and 0.65 for low-risk applications. As it is not too low in both cases,
there is not a large number of false negatives.

The F1 score, which is a summary statistic of precision and sensitivity, is really low for high-risk applications, which indicates an 
an imbalance between sensitivity and precision. 

In summary, this model may not be the best for predicting suitable loan applications.
Its accuracy of 0.62 is somewhat low, and the precision and recall are not good enough 
to state that it will be good at classifying high-risk applications.



### SMOTE

As modeling is an iterative process, we decided to sample the dataset this time using SMOTE, another oversampling 
approach to deal with unbalanced datasets.

![smote_resampling](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/smote_resample.png)

Then, we trained the Logistic Regression model using the resampled data and calculated the predictions:

![smote_predictions](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/smote_predictions.png)

To evaluate the performance of the model, we calculated the balanced accuracy score, and the confusion matrix and printed the imbalanced classification report:

![smote_accuracy](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/smote_accuracy.png)

![smote_sm](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/smote_cm.png)

![smote_report](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/smote_report.png)

The results show that:

* Out of 87 High-Risk applications, 56 were predicted to be High-Risk.
* Out of 87 High-Risk applications, 31 were predicted to be low-Risk.
* Out of 17118 low-Risk applications, 5840	 were predicted to be high-Risk.
* Out of 17118 low-Risk applications, 11278 were predicted to be low-Risk.

Looking at the report and the accuracy value, the metrics of the minority class(precision, recall, and F1 score) have
slightly improved. However, this model is still not the best at classifying high-risk applications.

### Random Undersampling

Another way to deal with class imbalance is to use undersampling, which is the opposite of oversampling. 
Instead of increasing the number of the minority class, the size of the majority class is decreased.

To proceed, we used the Cluster centroid undersampling algorithm and resampled the training data:

![cc_resample](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/cc_resample.png)

Then, we trained the Logistic Regression model using the resampled data and calculated the predictions:

![cc_predictions](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/cc_predictions.png)

 To evaluate the performance of the model, we calculated the balanced accuracy score, and the confusion matrix and printed the imbalanced classification report:
 
![cc_accuracy](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/cc_accuracy.png)

![cc_sm](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/cc_cm.png)

![cc_report](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/cc_report.png)

The results show that:

* Out of 87 High-Risk applications, 52 were predicted to be High-Risk.
* Out of 87 High-Risk applications, 35 were predicted to be low-Risk.
* Out of 17118 low-Risk applications, 9685 were predicted to be high-Risk.
* Out of 17118 low-Risk applications, 74333 were predicted to be low-Risk.

Looking at the report and the accuracy value, which decreased to 0.51, the metrics of the minority class(precision, recall, and F1 score) have
not improved. The metrics of the majority class have worsened. This model is still not the best at classifying high-risk applications.


### Combination (Over and Under) Sampling

The results of oversampling and undersampling are not satisfactory. So, we decided to try SMOTEENN, 
an approach to resampling that combines aspects of both oversampling and undersampling.

To proceed, we resampled the dataset:

![smoteenn_resample](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/smoteenn_resample.png)

Then, we trained the Logistic Regression model using the resampled data and calculated the predictions:

![smoteenn_predictions](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/smoteenn_predictions.png)

 To evaluate the performance of the model, we calculated the balanced accuracy score and the confusion matrix and printed the imbalanced classification report:
 
![smoteenn_accuracy](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/smoteenn_accuracy.png)

![smoteenn_report](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/smoteen_report2.png)

The results with SMOTEENN are similar to oversampling. The accuracy is 0.65, and the metrics of the minority class(precision, recall, and F1 score) are not
great, which indicates an imbalance between sensitivity and precision. This model is still not the best at classifying high-risk applications.

## Use Ensemble Classifiers to Predict Credit Risk

Since the results with Resampling Models to Predict Credit Risk are unsatisfactory, we decided to use ensemble classifiers.

The concept of ensemble learning is combining multiple models to help improve the accuracy and robustness and therefore increase the model's overall performance.

Similar to when using the Resampling Models, we prepared the data and split it into training and testing sets.

### Balanced Random Forest Classifier

The first ensemble algorithm we used is the Balanced Random Forest Classifier which randomly under-samples each bootstrap sample to balance it.

To proceed, we resampled the training data with the BalancedRandomForestClassifier:

![brf_resample](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/brf_resample.png)

Then, we calculate our predictions:

To evaluate the performance of the model, we calculated the balanced accuracy score and the confusion matrix and printed the imbalanced classification report:
 
![brf_accuracy](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/brf_accuracy.png)

![brf_sm](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/brf_cm.png)

![brf_report](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/brf_report.png)


The results show that:

* Out of 87 High-Risk applications, 30 were predicted to be High-Risk.
* Out of 87 High-Risk applications, 57 were predicted to be low-Risk.
* Out of 17118 low-Risk applications, 11 were predicted to be high-Risk.
* Out of 17118 low-Risk applications, 17107 were predicted to be low-Risk.

Looking at the report and the accuracy value, which is 0.67, our performance has increased. The metrics of the minority class(precision, recall, and F1 score) have
improved, and the metrics of the majority class have also improved. However,as the number of high-risk applications predicted as low-risk is high, this model might 
have a greater accuracy than the resampling ones, but it is still not the best to Predict Credit Risk.

### Easy Ensemble AdaBoost Classifier

We build our final model using Easy Ensemble AdaBoost Classifier.

To proceed, we resampled the training data with the EasyEnsembleClassifier:

![eec_resample](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/eec_resample.png)

After the data is resampled, we calculated the accuracy score of the model, generated a confusion matrix, and then printed out the imbalanced classification report:

![eec_accuracy](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/eec_accuracy.png)

![eec_report](https://github.com/valerielnd/Credit_Risk_Analysis/blob/main/Resources/eec_report.png)

The results show that:

* Out of 87 High-Risk applications, 79 were predicted to be High-Risk.
* Out of 87 High-Risk applications, 8 were predicted to be low-Risk.
* Out of 17118 low-Risk applications, 979 were predicted to be high-Risk.
* Out of 17118 low-Risk applications, 16139 were predicted to be low-Risk.

Looking at the report and the accuracy value, which is 0.92, the performance we obtained increased further. Some metrics of the minority class(precision and F1 score) have
worsened from the previous model, while the sensitivity has largely improved. The metrics of the majority class have slightly decreased. 
With a high accuracy 0f 0.92, this model seems to be the best one to Predict Credit Risk. It helps detect the greater high-risk applications out of the 87, contrary
to the previous models where not more than 50 were detected. Also, this model comes in the second position regarding the number of low-risk applications
detected as high-risk. The previous model made a better prediction, but considering the number of high-risk applications it predicted as low-risk, the last model might bring
less loss to the lending institution.

However, as the accuracy is somewhat high, this raises suspicion of overfitting. We could try evaluating this model further using another testing set. 