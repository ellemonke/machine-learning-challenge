# machine-learning-challenge
In this challenge, I used [Kepler Exoplanet Data](https://www.kaggle.com/nasa/kepler-exoplanet-search-results) to create a machine learning model.

#### Pre-Processing:
1. First, I cleaned the data using **Pandas and NumPy** and selected about 16 initial features to use in the model - all seemingly relevant to the target. 
2. I chose the column 'koi_disposition' with a classification of 'CONFIRMED' as the target, meaning the observed data would be classified as a 'Kepler Object of Interest' (KOI) or not. The target was binary encoded.
3. Using **SciKit Learn** from this point forward, I split the data into training and testing data and created scalers to transform them.

#### [Model 1 (16 features):](model_1.ipynb)
4. Using only a Logistic Regression model, the training data R<sup>2</sup> score was initially 0.762 and the testing data R<sup>2</sup> score was 0.763, with a MSE of 0.935.
5. Using K-Nearest Neighbors (KNN), I plotted a line chart to see at which value of k (number of data points in a cluster) would lead to the closest R<sup>2</sup> scores between the training and testing data. k=35 led to a training and testing R<sup>2</sup> score of 0.798. So far using all 16 features, this was the best model score which means there seemed to be some non-random clustering.
6. Using GridSearch, I discovered that parameters of C=1 and gamma=0.0001 resulted in a precision score 0.76; F1 score 0.74; and accuracy score 0.62. These were all lower than the KNN model.

#### [Model 2 (3 features):](model_2.ipynb)
4. It was possible that there were too many non-relevant features, so I used a Random Forest Classifier model to see if using the most important features would lead to a more accurate model. The top three features were 'koi_model_snr', 'koi_impact', and 'koi_prad'.
5. Using only a Logistic Regression Model, the training data R<sup>2</sup> score was then 0.743 and the testing data R<sup>2</sup> score was 0.742, with a MSE of 1.00. This just means that the model would have to be more complex than just three features or binary classification.
6. Using K-Nearest Neighbors, I found that k=45 led to a training and testing R<sup>2</sup> score of 0.819. This was slightly better than the 16-feature KNN model. 
7. Using GridSearch, I discovered that parameters of C=1 and gamma=0.0001 resulted in a precision score 0.74; F1 score 0.85; and accuracy score 0.74. Fine tuning the parameters led to a better F1 score, but there was still risk of false positives and the accuracy was still less than 0.8. So I chose the KNN model.

#### Final Choice: KNN using 3 features and k=45.
I chose the K-Nearest Neighbors model using only the feautures 'koi_model_snr', 'koi_impact', and 'koi_prad'. k=45 provided the best score (0.819) meaning clusters of about 45 tended to be similarly classified. I saved this model to [EllenHsu.sav](EllenHsu.sav) to be able to test future data against this model.
