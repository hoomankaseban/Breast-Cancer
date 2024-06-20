#!/usr/bin/env python
# coding: utf-8

# # IMPORT Libraries

# In[252]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# # READ CSV

# In[253]:


my_data = pd.read_csv("Breast_Cancer.csv", delimiter=",")
my_data[0:5]


# # Gain some Information

# In[254]:


my_data['Status'].unique()


# In[255]:


my_data.info()


# # Create DataFrame

# In[256]:


df=pd.DataFrame(my_data)


# # Information about quantity of values for each column

# In[257]:


for col in df.columns:
    if not pd.api.types.is_integer_dtype(df[col]):  # Check if column is not integer type
        print(f"Value counts for {col}:")
        print(df[col].value_counts())
        print()  # Print an empty line for separation


# # Gain some knowledge about numerical features

# In[258]:


numerical_columns = df.select_dtypes(include=['int', 'float']).columns

# Describe numerical columns
numerical_stats = df[numerical_columns].describe()

print("Descriptive statistics for numerical columns:")
print(numerical_stats)


# # Feature Selection

# In[259]:


df.info()


# # Feature selection by Descision Trees feature importance

# In[260]:


selected_columns = ['Age', 'Tumor Size', 'Regional Node Examined','Survival Months','Reginol Node Positive',]  # Replace with your actual feature names
# Separate features and target
X = df[selected_columns]
y = df['Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


# In[261]:


# Get feature importances
importances = clf.feature_importances_
importance_series = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print("Feature Importances from Decision Tree:")
print(importance_series)


# ## "Regional Node Examined" and "Reginol Node Positive" should be dropped

# In[262]:


categorical_columns = ['Status','Marital Status', 'differentiate','Race','T Stage ','Progesterone Status','N Stage','6th Stage','Grade','A Stage','Estrogen Status']  # Your actual feature names
X_columns = ['Marital Status', 'differentiate','Race','T Stage ','Progesterone Status','N Stage','6th Stage','Grade','A Stage','Estrogen Status']  # Your actual feature names

# Encode categorical features
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
                


# In[263]:


X = df[X_columns]
y = df['Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


# In[264]:


# Get feature importances
importances = clf.feature_importances_
importance_series = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print("Feature Importances from Decision Tree:")
print(importance_series)


# ## "Marital Status" and "6th stage" are selected

# # Selecting top features
# ## There are 3 numerical, 1 categorical, and 1 binary columns

# In[265]:


columns_to_drop=['T Stage ','N Stage','differentiate','Race','Grade','A Stage','Estrogen Status','Progesterone Status','Reginol Node Positive','Regional Node Examined']
df.drop(columns=columns_to_drop, inplace=True)  # Use inplace=True to modify the original DataFrame


# In[266]:


df


# # Categorizing Numerical features

# In[267]:


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


# In[268]:


df = categorize_numerical_feature(df, 'Age', 'Status')
df['Age_categorized'].value_counts()


# In[269]:


df = categorize_numerical_feature(df, 'Tumor Size', 'Status')
df['Tumor Size_categorized'].value_counts()


# In[270]:


df = categorize_numerical_feature(df, 'Survival Months', 'Status')
df['Survival Months_categorized'].value_counts()


# In[271]:


df


# In[272]:


df['Status'].value_counts()


# ### 0 means "Alive", 1 means "Dead"

# In[273]:


dead_count=int(df['Status'].value_counts().get(1,0))
alive_count=int(df['Status'].value_counts().get(0,0))
dataset_size=dead_count+alive_count
print(f'Dateset size is {dataset_size} \n{alive_count} are alive and {dead_count} are dead')


# # Calculate IG

# In[274]:


p_dead=dead_count/dataset_size
p_alive=1-p_dead
main_entropy=(-p_dead*np.log2(p_dead))-(p_alive*np.log2(p_alive))
main_entropy


# In[275]:


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


# In[276]:


df


# In[277]:


gain_age = information_gain(df['Status'], df['Age_categorized'])
gain_marital = information_gain(df['Status'], df['Marital Status'])
gain_6stage = information_gain(df['Status'], df['6th Stage'])
gain_month = information_gain(df['Status'], df['Survival Months_categorized'])
gain_tumor = information_gain(df['Status'], df['Tumor Size_categorized'])

print(f"Information Gains : \nAge={gain_age} , Marital status={gain_marital} \n6th stage={gain_6stage} , Survival months={gain_month} , Tumor size={gain_tumor}")


# # Building tree

# In[297]:


df1=df[['Status','Age_categorized','Marital Status','6th Stage','Survival Months_categorized','Tumor Size_categorized']].copy()


# In[298]:


df1


# In[314]:


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


# In[315]:


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


# In[316]:


tree = build_tree(df1, 'Status')


# In[301]:


tree


# # Predict

# In[318]:


def predict(tree, sample):
    # Traverse the tree until a leaf node is reached
    while isinstance(tree, dict):
        feature, subtree = next(iter(tree.items()))
        value = sample.get(feature)  # Get the value of the feature from the sample
        tree = subtree.get(value)  # Move to the next subtree based on the feature value

    return tree  # Return the predicted class label

# Example usage:
# Assume sample is a dictionary with feature values for a new instance
sample = {
    'Survival Months_categorized': '> 47.5',
    'Age_categorized': '> 61.5',
    '6th Stage': 0,
    'Tumor Size_categorized': '> 17.5',
    'Marital Status': 0
}

# Assuming 'tree' is already defined as your decision tree structure
prediction = predict(tree, sample)
print("Predicted class:", prediction)


# In[319]:


sample = {
    'Survival Months_categorized': '<= 47.5',
    'Age_categorized': '> 61.5',
    '6th Stage': 3,
    'Tumor Size_categorized': '> 17.5',
    'Marital Status': 4
}
prediction = predict(tree, sample)
print("Predicted class:", prediction)


# In[ ]:




