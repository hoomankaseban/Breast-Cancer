#!/usr/bin/env python
# coding: utf-8

# # IMPORT Libraries

# In[260]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# # READ CSV

# In[261]:


my_data = pd.read_csv("Breast_Cancer.csv", delimiter=",")
my_data[0:5]


# # Gain some Information

# In[262]:


my_data['Status'].unique()


# In[263]:


my_data.info()


# # Create DataFrame

# In[264]:


df=pd.DataFrame(my_data)


# # Information about quantity of values for each column

# In[265]:


for col in df.columns:
    if not pd.api.types.is_integer_dtype(df[col]):  # Check if column is not integer type
        print(f"Value counts for {col}:")
        print(df[col].value_counts())
        print()  # Print an empty line for separation


# # Gain knowledge about numerical features

# In[266]:


numerical_columns = df.select_dtypes(include=['int', 'float']).columns

# Describe numerical columns
numerical_stats = df[numerical_columns].describe()

print("Descriptive statistics for numerical columns:")
print(numerical_stats)


# # Feature selection by Descision Trees feature importance

# ### Selection in numerical features

# In[267]:


selected_columns = ['Age', 'Tumor Size', 'Regional Node Examined','Survival Months','Reginol Node Positive',]  # Replace with your actual feature names
# Separate features and target
X = df[selected_columns]
y = df['Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


# In[268]:


# Get feature importances
importances = clf.feature_importances_
importance_series = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print("Feature Importances from Decision Tree:")
print(importance_series)


# ## "Regional Node Examined" and "Reginol Node Positive" should be dropped

# ### Selection in categorical features

# In[269]:


categorical_columns = ['Status','Marital Status', 'differentiate','Race','T Stage ','Progesterone Status','N Stage','6th Stage','Grade','A Stage','Estrogen Status']  # Your actual feature names
X_columns = ['Marital Status', 'differentiate','Race','T Stage ','Progesterone Status','N Stage','6th Stage','Grade','A Stage','Estrogen Status']  # Your actual feature names

# Encode categorical features
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
                


# In[270]:


X = df[X_columns]
y = df['Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


# In[271]:


# Get feature importances
importances = clf.feature_importances_
importance_series = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print("Feature Importances from Decision Tree:")
print(importance_series)


# ## "Marital Status" and "6th stage" are selected

# # Selecting top features
# ## There are 3 numerical, 1 categorical, and 1 binary columns

# In[272]:


columns_to_drop=['T Stage ','N Stage','differentiate','Race','Grade','A Stage','Estrogen Status','Progesterone Status','Reginol Node Positive','Regional Node Examined']
df.drop(columns=columns_to_drop, inplace=True)  # Use inplace=True to modify the original DataFrame


# In[273]:


df


# # Categorizing Numerical features

# In[274]:


def entropy(y):
    unique_classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

# Function to calculate information gain
def information_gain(y, split_point, feature_values):
    left_indices = feature_values <= split_point
    right_indices = feature_values > split_point
    left_entropy = entropy(y[left_indices])
    right_entropy = entropy(y[right_indices])
    p_left = len(y[left_indices]) / len(y)
    p_right = len(y[right_indices]) / len(y)
    return entropy(y) - (p_left * left_entropy + p_right * right_entropy)

# Function to find the best split point for a numerical feature
def find_best_split_point(y, feature_values):
    unique_values = np.sort(np.unique(feature_values))
    best_split = None
    best_gain = -1
    for i in range(len(unique_values) - 1):
        split_point = (unique_values[i] + unique_values[i + 1]) / 2
        gain = information_gain(y, split_point, feature_values)
        if gain > best_gain:
            best_gain = gain
            best_split = split_point
    return best_split, best_gain

# Function to categorize a numerical feature
def categorize_numerical_feature(df, feature_name, target_name):
    feature_values = df[feature_name].values
    target_values = df[target_name].values
    split_point, _ = find_best_split_point(target_values, feature_values)
    df[f'{feature_name}_categorized'] = pd.cut(feature_values, bins=[-np.inf, split_point, np.inf], labels=[f'<= {split_point}', f'> {split_point}'])
    return df


# In[275]:


df = categorize_numerical_feature(df, 'Age', 'Status')
df['Age_categorized'].value_counts()


# In[276]:


df = categorize_numerical_feature(df, 'Tumor Size', 'Status')
df['Tumor Size_categorized'].value_counts()


# In[277]:


df = categorize_numerical_feature(df, 'Survival Months', 'Status')
df['Survival Months_categorized'].value_counts()


# In[278]:


df


# ### 0 means "Alive", 1 means "Dead"

# In[279]:


dead_count=int(df['Status'].value_counts().get(1,0))
alive_count=int(df['Status'].value_counts().get(0,0))
dataset_size=dead_count+alive_count
print(f'Dateset size is {dataset_size} \n{alive_count} are alive and {dead_count} are dead')


# # Spliting

# In[280]:


df1=df[['Status','Age_categorized','Marital Status','6th Stage','Survival Months_categorized','Tumor Size_categorized']].copy()


# In[281]:


df1


# In[282]:


X = df1[['Age_categorized','Marital Status','6th Stage','Survival Months_categorized','Tumor Size_categorized']]
X[0:5]


# In[283]:


y=df1['Status']
y.value_counts()


# In[284]:


trainset, testset = train_test_split(df1, test_size=0.3, random_state=3)


# In[285]:


print(f'Train set size :{len(trainset)}\nTest set size :{len(testset)}')


# # Building tree

# ### Functions to Calculate IG

# In[286]:


def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(y, feature_values):
    total_entropy = entropy(y)
    weighted_entropy = 0
    unique_values = np.unique(feature_values)
    
    for value in unique_values:
        subset_indices = feature_values == value
        subset_entropy = entropy(y[subset_indices])
        weighted_entropy += (np.sum(subset_indices) / len(y)) * subset_entropy
        
    return total_entropy - weighted_entropy


# ### Function to select the best feature to splite

# In[287]:


def best_feature_to_split(df, target_name):
    features = df.columns.drop(target_name)
    best_gain = -1
    best_feature = None
    
    for feature in features:
        gain = information_gain(df[target_name], df[feature])
        if gain > best_gain:
            best_gain = gain
            best_feature = feature
            
    return best_feature


# ### Main function to build the tree

# In[288]:


def build_tree(df, target_name):
    # If all target values are the same, return a leaf node
    if len(np.unique(df[target_name])) == 1:
        return df[target_name].iloc[0]
    
    # If there are no more features to split on, return the most common target value
    if len(df.columns) == 1:
        return df[target_name].mode()[0]
    
    # Choose the best feature to split on
    best_feature = best_feature_to_split(df, target_name)
    if best_feature is None:
        return df[target_name].mode()[0]
    
    # Create the tree structure
    tree = {best_feature: {}}
    unique_values = np.unique(df[best_feature])
    
    # Split the dataset and recursively build subtrees
    for value in unique_values:
        subset = df[df[best_feature] == value].drop(columns=[best_feature])
        subtree = build_tree(subset, target_name)
        tree[best_feature][value] = subtree
        
    return tree


# In[289]:


tree = build_tree(trainset, 'Status')


# In[290]:


tree


# # Predict

# In[291]:


y_testset=testset['Status'].copy()
y_testset


# In[295]:


x_testset=testset.drop(columns=['Status'])
x_testset


# In[305]:


def predict_single(tree, sample):
    while isinstance(tree, dict):
        feature = next(iter(tree))
        value = sample[feature]
        tree = tree[feature].get(value)
        if tree is None:  # Handle cases where the value is not in the tree
            return None  # Or some default value
    return tree

def dataframe_predict(tree, test_set):
    predictions = test_set.apply(lambda x: predict_single(tree, x), axis=1)
    return predictions


# In[304]:


sample = {
    'Survival Months_categorized': '> 47.5',
    'Age_categorized': '> 61.5',
    '6th Stage': 0,
    'Tumor Size_categorized': '> 17.5',
    'Marital Status': 0
}
sample2 = {
    'Survival Months_categorized': '<= 47.5',
    'Age_categorized': '> 61.5',
    '6th Stage': 3,
    'Tumor Size_categorized': '> 17.5',
    'Marital Status': 4
}
sample3 = {
    'Survival Months_categorized': '> 47.5',
    'Age_categorized': '> 61.5',
    '6th Stage': 2,
    'Tumor Size_categorized': '> 17.5',
    'Marital Status': 1
}
# Assuming 'tree' is already defined as your decision tree structure
prediction1 = predict_single(tree, sample)
prediction2 = predict_single(tree, sample2)
prediction3 = predict_single(tree, sample3)

print(f"Predicted class for sample 1: {prediction1}\nPredicted class for sample 2: {prediction2}\nPredicted class for sample 3: {prediction3}")


# # Evaluation

# In[319]:


predictions = dataframe_predict(tree, testset)
predictions_list = predictions.tolist()
predictions_list[:5]


# In[317]:


y_testset=y_testset.tolist()
predictions_list = predictions.tolist()
y_testset[:5]


# In[320]:


for i in range(5):
    if not(predictions_list[i]==1):
        print (y_testset[i])


# In[324]:


def Accuracy(y_pred,y_actual):
    T_np=0
    F_np=0
    for i in range(len(y_pred)):
        if y_pred[i]==y_actual[i]:
            T_np+=1
        else:
            F_np+=1
    acc=T_np/(T_np+F_np)
    return acc


# In[325]:


acc=Accuracy(predictions_list,y_testset)
acc


# In[ ]:




