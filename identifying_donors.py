import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
import visuals as vs

# Pretty display for notebooks
%matplotlib inline

# Load the Census dataset
data = pd.read_csv(r"C:\Users\wsoo\Documents\GitHub\machine-learning\projects\2. finding_donors\census.csv")


#understanding the data
print ("Data frame has {} rows and {} columns.".format(*data.shape))
data.head(5)

'''
Data exploration
'''

#Number of records where individual's income is more than $50,000
data['income'] 
data['income'] == '>50K' #not very useful

data[data['income'] == '>50K'] 
data[data['income'] == '>50K'].shape
len(data[data['income'] == '>50K'])


#Total number of records
n_records = len(data['income'])


#Number of records where individual's income is more than $50,000
n_greater_50k = len(data[data['income'] == '>50K'])

#Number of records where individual's income is at most $50,000
n_at_most_50k = len(data[data['income'] == '<=50K'])

#Percentage of individuals whose income is more than $50,000
greater_percent = n_greater_50k/n_records*100

# Print the results
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {:.3}%".format(greater_percent))

'''
Trnasforming skewed data
'''

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
vs.distribution(data)

'''
Log-transform the skewed features
'''

skewed = ['capital-gain', 'capital-loss']
#getting the featured columns
features_log_transformed = pd.DataFrame(data = features_raw)
#LOG-transfor the skewed features ONLY
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

'''
LATEST DATAFRAME AT THIS POINT IS features_log_transformed
'''


# Visualize the new log distributions
vs.distribution(features_log_transformed, transformed = True)


'''
Normalize data - for columns which takes numerical values
'''
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Initialize a scaler, then apply it to the features
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

#recreating another set of dataframe
features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
#only normalizing the numerical columns 
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])
# Show an example of a record with scaling applied
display(features_log_minmax_transform.head(n = 5))



'''
Data processing - one hot encoding and changing target to 1 and 0
'''
list(data.columns.values)
print (type(data.columns.values))
categorical = ['workclass','education_level', 'marital-status', 'occupation', 'relationship', \
              'race', 'sex' , 'native-country']

#One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
features_final = pd.get_dummies(features_log_minmax_transform)

# Encode the 'income_raw' data to numerical values
income = income_raw.map({'<=50K': 0 , '>50K' : 1})

print(income)
# Print the number of features after one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Uncomment the following line to see the encoded feature names
# print encoded

'''
Shuffle and Split data
'''
# Import train_test_split
from sklearn.model_selection import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)


'''
Evaluating Model performance
'''


TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data 
#encoded to numerical values done in the data preprocessing step.
FP = income.count() - TP # Specific to the naive case

TN = 0 # No predicted negatives in the naive case
FN = 0 # No predicted negatives in the naive case

# Calculate accuracy, precision and recall
accuracy = TP/(TP+FP)*100
recall = TP/(TP+FP)*100
precision = TP/ (TP+FN) *100

# Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
beta_square = 0.5**2
fscore = (1 + beta_square) * (precision * recall) / ((beta_square * precision) + recall)

# Print the results 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))


'''
Model to Choose

1. Decision tree
2. Naive Bayes - probablistic approach
3. KNN - problem = need to choose the number of neighbours
if not
4. Random Forest 

SVM not good since there are large number of features.

'''


'''
Creating Training and Prediction Pipeline
'''
time()

# Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])                                                                           #double check this later
    end = time() # Get end time
    
    #Calculate the training time
    results['train_time'] = end - start
        
    # Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])                                                                      #double check this later
    end = time() # Get end time
    
    # Calculate the total prediction time
    results['pred_time'] = end - start
            
    # Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, average='weighted', beta=0.5)
        
    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, average='weighted', beta=0.5)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results
    
    
'''
Model Implemenatation
'''

#Import the three supervised learning models from sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# TODO: Initialize the three models
clf_A = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
clf_B = GaussianNB()
clf_C = KNeighborsClassifier()

#Calculate the number of samples for 1%, 10%, and 100% of the training data
# HINT: samples_100 is the entire training set i.e. len(y_train)
# HINT: samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
# HINT: samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
samples_100 = len(X_train)
samples_10 = int(len(X_train)*0.1)
samples_1 = int(len(X_train)*0.01)

# Collect results on the learners
results = {}

print 

for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
        train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)

'''
Knn out because it takes too long and it seems that it is as good as Decision Tree. 
For large number of training set, it seems Decision Tree and Naive Bayes are equally good. Time taken to run these 2 classifiers are short. 
'''


'''
Qn: Which model to choose?
Ans: From the above graph, the most ideal classifier is Decision tree.  
'''




'''
To execute parameter tuning after model selected
'''

#Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.metrics import  make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score


# TODO: Initialize the classifier
clf = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

# Create the parameters list you wish to tune, using a dictionary if needed.
# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
parameters = {'max_depth': [i for i in range(1,11)], 'criterion': ['gini', 'entropy']}

#Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(score_func = fbeta_score, beta = 0.5 )

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(estimator = clf, param_grid = parameters, scoring = scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Get the estimator
best_clf = grid_fit.best_estimator_
#show the params in the classifier
print(best_clf) 

# Make predictions using the unoptimized and model
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)



# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))


'''
An important task when performing supervised learning on a dataset like the census data we study here is determining which features provide 
the most predictive power. 

By focusing on the relationship between only a few crucial features and the target label we simplify our understanding of the phenomenon,
which is most always a useful thing to do. In the case of this project, that means we wish to identify a small number of features that 
most strongly predict whether an individual makes at most or more than $50,000.

A scikit-learn classifier (e.g., adaboost, random forests) that has a feature_importance_ attribute, which is a function that ranks the importance of features according to the chosen classifier. 
To determine the top 5 most important features for the census dataset.
'''

from sklearn.ensemble import AdaBoostClassifier

clf_ADA = AdaBoostClassifier()
clf_ADA.fit(X_train, y_train)

#To show the important features.
len(clf_ADA.feature_importances_)

ADA_pred = clf_ADA.predict(X_train)



'''
Another way of extracting important features
'''
# Import a supervised learning model that has 'feature_importances_'
 from sklearn.ensemble import GradientBoostingClassifier

# Train the supervised model on the training set using .fit(X_train, y_train)
model = GradientBoostingClassifier().fit(X_train, y_train)

# Extract the feature importances using .feature_importances_ 
importances = model.feature_importances_ 

# Plot
vs.feature_plot(importances, X_train, y_train)



'''
Using the model used above on the top 5 features!!
'''

# Import functionality for cloning a model
from sklearn.base import clone

# Reduce the feature space
X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

# Train on the "best" model found from grid search earlier
clf = (clone(best_clf)).fit(X_train_reduced, y_train)

# Make new predictions
reduced_predictions = clf.predict(X_test_reduced)

# Report scores from the final model using both versions of data
print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))

'''
This show that reducing the number of attributes to 5 marginally/barely reduced the F score
'''