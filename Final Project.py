#!/usr/bin/env python
# coding: utf-8

# # Final Project - Programming II
# ## Maria Veronica Ortega Lacruz
# ### December, 12th 2024

# ---

# In[9]:


### Q1. Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

import pandas as pd

### Read the CVS file
s = pd.read_csv('/Users/mveronicaol/Programming 2-Georgetown/social_media_usage.csv')

### Check the dimension of the dataframe
print(s.shape)


# ---

# In[11]:


### Q2. Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1.
### If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and 
### two columns and test your function to make sure it works as expected. 

import numpy as np
import pandas as pd

# Define the clean_sm function
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Create a toy dataframe
toy_data = pd.DataFrame({
    'column1': [1, 2, 1],
    'column2': [0, 1, 3]
})

# Apply the function to all columns of the toy dataframe
toy_data_cleaned = toy_data.applymap(clean_sm)

print(toy_data_cleaned)


# ---

# In[13]:


### Q3. Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary 
### variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not 
### the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), 
### education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and 
### age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features 
### are related to the target.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the clean_sm function
def clean_sm(x):
    return np.where(x == 1, 1, 0)

# Clean and prepare data for the new dataframe
s['sm_li'] = clean_sm(s['web1h'])  # Assuming 'web1h' indicates LinkedIn usage
s['income'] = np.where(s['income'] <= 9, s['income'], np.nan)
s['education'] = np.where(s['educ2'] <= 8, s['educ2'], np.nan)
s['age'] = np.where(s['age'] <= 98, s['age'], np.nan)
s['parent'] = clean_sm(s['par'])
s['married'] = clean_sm(s['marital'])
s['female'] = clean_sm(s['gender'] == 1)  # Assuming 1 = Female, 2 = Male

# Create the new dataframe and drop missing values
ss = s[['income', 'education', 'parent', 'married', 'female', 'age', 'sm_li']].dropna()

# Perform exploratory analysis
print(ss.describe(include='all'))

# Visualize the distribution of each feature
for column in ['income', 'education', 'age']:
    plt.figure()
    plt.hist(ss[column], bins=10, edgecolor='black')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Visualize the relationship between the target and binary features
for column in ['parent', 'married', 'female']:
    plt.figure()
    ss.groupby(column)['sm_li'].mean().plot(kind='bar')
    plt.title(f'Average LinkedIn Usage by {column}')
    plt.ylabel('Proportion of LinkedIn Users')
    plt.show()


# ---

# In[15]:


### Q4. Create a target vector (y) and feature set (X)

# Define the target vector (y) and feature set (X)
y = ss['sm_li']  # Target vector: LinkedIn usage
X = ss.drop(columns=['sm_li'])  # Feature set: All other columns

# Display the dimensions of X and y
print(X.shape, y.shape)


# ---

# In[17]:


### Q5. Split the data into training and test sets. Hold out 20% of the data for testing. 
### Explain what each new object contains and how it is used in machine learning

from sklearn.model_selection import train_test_split

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Display the dimensions of the resulting datasets
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# The dataset was split into training and test sets to enable effective machine learning model training and evaluation. 
# The training set consists of 80% of the original data, with the features (X_train) and corresponding target values (y_train) 
# used to train the model. This allows the model to learn patterns and relationships between the input features and the target 
# variable (LinkedIn usage). The remaining 20% of the data is set aside as the test set, which includes the features (X_test) 
# and the true target values (y_test). The test set is used to evaluate the model's performance on unseen data, ensuring that 
# the model generalizes well to new inputs. By holding out a portion of the data for testing, we can assess the model's accuracy 
# and robustness in making predictions.


# ---

# In[19]:


### Q6. Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

from sklearn.linear_model import LogisticRegression

# Instantiate the logistic regression model with class_weight set to 'balanced'
log_reg = LogisticRegression(class_weight='balanced', random_state=42)

# Fit the model to the training data
log_reg.fit(X_train, y_train)

# Display the coefficients and intercept of the model
print(log_reg.coef_, log_reg.intercept_)


# ---

# In[21]:


### Q7. Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions 
### and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

from sklearn.metrics import accuracy_score, confusion_matrix

# Make predictions on the test data
y_pred = log_reg.predict(X_test)

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)


# The model achieved an accuracy of 64.68% on the test data, meaning it correctly predicted LinkedIn usage for approximately 65% of cases.
# The confusion matrix provides further insights into the model's performance. It correctly identified 104 individuals as not using LinkedIn
# (true negatives) and 59 individuals as using LinkedIn (true positives). However, the model incorrectly predicted 64 individuals as using 
# LinkedIn when they do not (false positives) and failed to identify 25 individuals who actually use LinkedIn (false negatives). 
# This highlights the model's strength in predicting non-users but also indicates some challenges in accurately identifying all users. 
# These insights can guide further model improvements.


# --- 

# In[23]:


### Q8. Create the confusion matrix as a dataframe and add informative column names and index names that indicate what
## each quadrant represents

import pandas as pd
from sklearn.metrics import confusion_matrix

# Assuming `y_test` and `y_pred` are already defined
conf_matrix = confusion_matrix(y_test, y_pred)

# Create the confusion matrix as a dataframe with informative labels
conf_matrix_df = pd.DataFrame(
    conf_matrix,
    columns=["Predicted: Not LinkedIn User", "Predicted: LinkedIn User"],
    index=["Actual: Not LinkedIn User", "Actual: LinkedIn User"]
)

print(conf_matrix_df)


# ---

# In[26]:


### Q9. Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. 
### Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual 
### example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report 
### using sklearn and check to ensure your metrics match those of the classification_report.

from sklearn.metrics import confusion_matrix, classification_report

# Use the existing predictions and true labels to generate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Extract values from the confusion matrix
tn, fp, fn, tp = conf_matrix.ravel()

# Calculate precision, recall, and F1 score by hand
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Print the metrics calculated by hand
print(f"Precision (by hand): {precision:.2f}")
print(f"Recall (by hand): {recall:.2f}")
print(f"F1 Score (by hand): {f1_score:.2f}")

# Generate the classification report using sklearn
class_report = classification_report(y_test, y_pred, target_names=["Not LinkedIn User", "LinkedIn User"])

# Print the classification report
print("\nClassification Report:\n", class_report)


# In[28]:


### Q10. Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high 
### level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change 
### if another person is 82 years old, but otherwise the same?

import pandas as pd

# Create feature sets for the two individuals
individual_1 = pd.DataFrame({
    'income': [8],
    'education': [7],
    'parent': [0],
    'married': [1],
    'female': [1],
    'age': [42]
})

individual_2 = pd.DataFrame({
    'income': [8],
    'education': [7],
    'parent': [0],
    'married': [1],
    'female': [1],
    'age': [82]
})

# Use the trained logistic regression model to predict probabilities
prob_1 = log_reg.predict_proba(individual_1)[0][1]
prob_2 = log_reg.predict_proba(individual_2)[0][1]

print(f"Probability for individual 1 (42 years old): {prob_1:.2f}")
print(f"Probability for individual 2 (82 years old): {prob_2:.2f}")


# ---

# In[ ]:




