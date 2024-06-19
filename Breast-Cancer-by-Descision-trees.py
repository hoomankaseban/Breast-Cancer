#!/usr/bin/env python
# coding: utf-8

# # IMPORT Libraries

# In[124]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# # READ CSV

# In[125]:


my_data = pd.read_csv("Breast_Cancer.csv", delimiter=",")
my_data[0:5]


# # Gain some Information

# In[126]:


my_data['Status'].unique()


# In[127]:


my_data.info()


# # Create DataFrame

# In[128]:


df=pd.DataFrame(my_data)


# # Information about quantity of values for each column

# In[129]:


for col in df.columns:
    if not pd.api.types.is_integer_dtype(df[col]):  # Check if column is not integer type
        print(f"Value counts for {col}:")
        print(df[col].value_counts())
        print()  # Print an empty line for separation


# # Gain some knowledge about numerical features

# In[130]:


numerical_columns = df.select_dtypes(include=['int', 'float']).columns

# Describe numerical columns
numerical_stats = df[numerical_columns].describe()

print("Descriptive statistics for numerical columns:")
print(numerical_stats)


# # Feature Selection

# In[131]:


df.info()


# # Feature selection by Descision Trees feature importance

# In[132]:


selected_columns = ['Age', 'Tumor Size', 'Regional Node Examined','Survival Months','Reginol Node Positive',]  # Replace with your actual feature names
# Separate features and target
X = df[selected_columns]
y = df['Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


# In[133]:


# Get feature importances
importances = clf.feature_importances_
importance_series = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print("Feature Importances from Decision Tree:")
print(importance_series)


# ## "Regional Node Examined" and "Reginol Node Positive" should be dropped

# In[134]:


categorical_columns = ['Marital Status', 'differentiate','Race','T Stage ','Progesterone Status','N Stage','6th Stage','Grade','A Stage','Estrogen Status']  # Your actual feature names

# Encode categorical features
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
                


# In[135]:


X = df[categorical_columns]
y = df['Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Decision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)


# In[136]:


# Get feature importances
importances = clf.feature_importances_
importance_series = pd.Series(importances, index=X.columns).sort_values(ascending=False)

print("Feature Importances from Decision Tree:")
print(importance_series)


# ## "Marital Status" and "6th stage" are selected

# # Selecting top features
# ## There are 3 numerical, 1 categorical, and 1 binary columns

# In[137]:


columns_to_drop=['T Stage ','N Stage','differentiate','Race','Grade','A Stage','Estrogen Status','Progesterone Status','Reginol Node Positive','Regional Node Examined']
df.drop(columns=columns_to_drop, inplace=True)  # Use inplace=True to modify the original DataFrame


# In[138]:


df


# In[ ]:




