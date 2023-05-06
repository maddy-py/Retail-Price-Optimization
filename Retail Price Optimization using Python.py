#!/usr/bin/env python
# coding: utf-8

# # Retail Price Optimization using Python

# Retail price optimization involves determining the optimal selling price for products or services to maximize revenue and profit. So, if you want to learn how to use machine learning for the retail price optimization task, this article is for you. In this article, I will walk you through the task of Retail Price Optimization with Machine Learning using Python.

# What is Retail Price Optimization?
# Optimizing retail prices means finding the perfect balance between the price you charge for your products and the number of products you can sell at that price.
# 
# The ultimate aim is to charge a price that helps you make the most money and attracts enough customers to buy your products. It involves using data and pricing strategies to find the right price that maximizes your sales and profits while keeping customers happy.
# 
# So for the task of Retail Price Optimization, you need data about the prices of products or services and everything that affects the price of a product. I found an ideal dataset for this task. You can download the data from here.
# 
# In the section below, I will take you through the task of Retail Price Optimization with Machine Learning using Python.

# Let’s start the task of Retail Price Optimization by importing the necessary Python libraries and the dataset:

# In[1]:


import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"


# In[2]:


data = pd.read_csv('retail_price.csv')
print(data.head())


# Before moving forward, let’s have a look if the data has null values or not:

# In[3]:


print(data.isnull().sum())


# Now let’s have a look at the descriptive statistics of the data:

# In[4]:


print(data.describe())


# Now let’s have a look at the distribution of the prices of the products:

# In[5]:


fig = px.histogram(data, 
                   x='total_price', 
                   nbins=20, 
                   title='Distribution of Total Price')
fig.show()


# Now let’s have a look at the distribution of the unit prices using a box plot:

# In[6]:


fig = px.box(data, 
             y='unit_price', 
             title='Box Plot of Unit Price')
fig.show()


# Now let’s have a look at the relationship between quantity and total prices:

# In[7]:


fig = px.scatter(data, 
                 x='qty', 
                 y='total_price', 
                 title='Quantity vs Total Price', trendline="ols")
fig.show()

Thus, the relationship between quantity and total prices is linear. It indicates that the price structure is based on a fixed unit price, where the total price is calculated by multiplying the quantity by the unit price.
# Now let’s have a look at the average total prices by product categories:

# In[8]:


fig = px.bar(data, x='product_category_name', 
             y='total_price', 
             title='Average Total Price by Product Category')
fig.show()


# Now let’s have a look at the distribution of total prices by weekday using a box plot:

# In[9]:


fig = px.box(data, x='weekday', 
             y='total_price', 
             title='Box Plot of Total Price by Weekday')
fig.show()


# Now let’s have a look at the distribution of total prices by holiday using a box plot:

# In[10]:


fig = px.box(data, x='holiday', 
             y='total_price', 
             title='Box Plot of Total Price by Holiday')
fig.show()


# Now let’s have a look at the correlation between the numerical features with each other:

# In[11]:


correlation_matrix = data.corr()
fig = go.Figure(go.Heatmap(x=correlation_matrix.columns, 
                           y=correlation_matrix.columns, 
                           z=correlation_matrix.values))
fig.update_layout(title='Correlation Heatmap of Numerical Features')
fig.show()


# In[12]:


data['comp_price_diff'] = data['unit_price'] - data['comp_1'] 

avg_price_diff_by_category = data.groupby('product_category_name')['comp_price_diff'].mean().reset_index()

fig = px.bar(avg_price_diff_by_category, 
             x='product_category_name', 
             y='comp_price_diff', 
             title='Average Competitor Price Difference by Product Category')
fig.update_layout(
    xaxis_title='Product Category',
    yaxis_title='Average Competitor Price Difference'
)
fig.show()


# Retail Price Optimization Model with Machine Learning
Now let’s train a Machine Learning model for the task of Retail Price Optimization. Below is how we can train a Machine Learning model for this problem:
# In[13]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

X = data[['qty', 'unit_price', 'comp_1', 
          'product_score', 'comp_price_diff']]
y = data['total_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2,
                                                    random_state=42)

# Train a linear regression model
model = DecisionTreeRegressor()
model.fit(X_train, y_train)


# In[14]:


y_pred = model.predict(X_test)

fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', 
                         marker=dict(color='blue'), 
                         name='Predicted vs. Actual Retail Price'))
fig.add_trace(go.Scatter(x=[min(y_test), max(y_test)], y=[min(y_test), max(y_test)], 
                         mode='lines', 
                         marker=dict(color='red'), 
                         name='Ideal Prediction'))
fig.update_layout(
    title='Predicted vs. Actual Retail Price',
    xaxis_title='Actual Retail Price',
    yaxis_title='Predicted Retail Price'
)
fig.show()


# So this is how you can optimize retail prices with Machine Learning using Python.

# Summary

# The ultimate aim of optimizing retail prices is to charge a price that helps you make the most money and attracts enough customers to buy your products. It involves using data and pricing strategies to find the right price that maximizes your sales and profits while keeping customers happy. I hope you liked this article on optimizing retail prices with Machine Learning using Python. Feel free to ask valuable questions in the comments section below.

# In[ ]:




