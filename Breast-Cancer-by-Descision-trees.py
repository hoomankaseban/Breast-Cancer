#!/usr/bin/env python
# coding: utf-8

# # IMPORT Libraries

# In[403]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# # READ CSV

# In[404]:


my_data = pd.read_csv("Breast_Cancer.csv", delimiter=",")
my_data[0:5]


# # Gain some Information

# In[405]:


my_data['Status'].unique()


# In[406]:


my_data.info()


# # Create DataFrame

# In[407]:


df=pd.DataFrame(my_data)


# #### Information about quantity of values for each column

# In[408]:


for col in df.columns:
    if not pd.api.types.is_integer_dtype(df[col]):  # Check if column is not integer type
        print(f"Value counts for {col}:")
        print(df[col].value_counts())
        print()  # Print an empty line for separation


# ### Gain knowledge about numerical features

# In[409]:


numerical_columns = df.select_dtypes(include=['int', 'float']).columns

# Describe numerical columns
numerical_stats = df[numerical_columns].describe()

print("Descriptive statistics for numerical columns:")
print(numerical_stats)


# ## *
# ## *
# ## *
# # Label Encoding

# In[410]:


df


# In[411]:


categorical_columns = ['Status','Marital Status', 'differentiate','Race','T Stage ','Progesterone Status','N Stage','6th Stage','Grade','A Stage','Estrogen Status']  # Your actual feature names
# Encode categorical features
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
df


# ## *
# ## *
# ## *
# # Feature selection by Mutual Information

# In[412]:


def entropy(labels):
    """Compute entropy of a list of labels."""
    unique, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    return -np.sum(probabilities * np.log2(probabilities))

def mutual_information(X, y):
    """Compute mutual information between feature matrix X and target vector y."""
    n_features = X.shape[1]
    mi_scores = np.zeros(n_features)
    
    # Compute entropy of the target variable y
    H_y = entropy(y)
    
    for i in range(n_features):
        feature_values = X[:, i]
        
        # Calculate entropy of the feature
        H_feature = entropy(feature_values)
        
        # Calculate joint entropy between the feature and target
        joint_entropy = 0
        unique_feature_values = np.unique(feature_values)
        for value in unique_feature_values:
            y_given_feature = y[feature_values == value]
            prob_y_given_feature = len(y_given_feature) / len(y)
            H_y_given_feature = entropy(y_given_feature)
            joint_entropy += prob_y_given_feature * H_y_given_feature
        
        # Calculate mutual information
        mi_scores[i] = H_feature - joint_entropy / H_y
    
    return mi_scores


# In[413]:


y=df[['Status']].values
y


# ## Selection in numerical features

# In[414]:


x_numerical=df[['Age', 'Tumor Size', 'Regional Node Examined','Survival Months','Reginol Node Positive']].values
x_numerical[:5]


# In[415]:


mi_scores = mutual_information(x_numerical, y)
print("Mutual Information Scores:", mi_scores)


# ### "Regional Node Examined" and "Reginol Node Positive" should be dropped

# ## Selection in categorical features

# In[416]:


x_categorical=df[['Marital Status', 'differentiate','Race','T Stage ','Progesterone Status','N Stage','6th Stage','Grade','A Stage','Estrogen Status']].values
x_categorical[:5]


# In[417]:


mi_scores = mutual_information(x_categorical, y)
print("Mutual Information Scores:", mi_scores)


# ### "Marital Status" and "6th stage" are selected
# ## *
# ## *
# ## *
# 

# # Selecting top features
# ## There are 3 numerical, 1 categorical, and 1 binary columns

# In[418]:


columns_to_drop=['T Stage ','N Stage','differentiate','Race','Grade','A Stage','Estrogen Status','Progesterone Status','Reginol Node Positive','Regional Node Examined']
df.drop(columns=columns_to_drop, inplace=True)  # Use inplace=True to modify the original DataFrame


# In[419]:


df


# ## *
# ## *
# ## *
# # Categorizing Numerical features

# In[420]:


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


# In[421]:


df = categorize_numerical_feature(df, 'Age', 'Status')
df['Age_categorized'].value_counts()


# In[422]:


df = categorize_numerical_feature(df, 'Tumor Size', 'Status')
df['Tumor Size_categorized'].value_counts()


# In[423]:


df = categorize_numerical_feature(df, 'Survival Months', 'Status')
df['Survival Months_categorized'].value_counts()


# In[424]:


df


# ### 0 means "Alive", 1 means "Dead"

# ## *
# ## *
# ## *
# # Splitter

# In[425]:


df1=df[['Status','Age_categorized','Marital Status','6th Stage','Survival Months_categorized','Tumor Size_categorized']].copy()


# In[426]:


df1


# In[427]:


trainset, testset = train_test_split(df1, test_size=0.3, random_state=3)


# In[428]:


print(f'Train set size :{len(trainset)}\nTest set size :{len(testset)}')


# ## *
# ## *
# ## *
# # Building descision tree by ID3

# ### Functions to Calculate IG

# In[429]:


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

# In[430]:


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

# In[431]:


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


# In[432]:


tree = build_tree(trainset, 'Status')


# In[433]:


tree


# # Predict

# In[434]:


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


# In[435]:


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


# ## *
# ## *
# ## *
# # Evaluation by Accuracy method

# In[438]:


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


# In[436]:


y_testset=testset['Status'].copy()
x_testset=testset.drop(columns=['Status'])
y_testset[:5]


# In[437]:


predictions = dataframe_predict(tree, testset)
predictions[:5]


# In[439]:


y_testset=y_testset.tolist()
predictions=predictions.tolist()
acc=Accuracy(predictions,y_testset)
acc


# # Accuracy of ID3 algorithm is 90 %

# ## *
# ## *
# ## *
# ## *
# # Cart and C4.5 algorithms...

# In[ ]:




