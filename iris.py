import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load the iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Target
df['target'] = iris.target.astype(str)
# df['target'] = df['target'].astype(int)

# Convert target to target names
df.loc[df['target'] == '0', 'target'] = 'setosa'
df.loc[df['target'] == '1', 'target'] = 'versicolor'
df.loc[df['target'] == '2', 'target'] = 'virginica'

# Build the model
x = iris.data[:, [0, 2]]
y = iris.target

# Logistic Regression
model = LogisticRegression()
model.fit(x, y)

# Sidebar
st.sidebar.header('Input Features')
sepal_length = st.sidebar.slider('Sepal Length (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.sidebar.slider('Petal Length (cm)', min_value=0.0, max_value=10.0, step=0.1)

# Main Panel
st.title('Iris Flower Prediction')
st.write('This app predicts the type of Iris flower based on the input features.')

# Input data
value_df = pd.DataFrame([], columns=['data','sepal_length (cm)', 'petal_length (cm)'])
record = pd.Series(['data', sepal_length, petal_length], index=value_df.columns)
value_df = pd.concat([value_df, record.to_frame().T], ignore_index=True)
value_df.set_index('data', inplace=True)
st.write('Input Data')

# Display the input data
st.write(value_df)

# Prediction
pred_probs = model.predict_proba(value_df)
pred_df = pd.DataFrame(pred_probs, columns=iris.target_names, index=['Probability'])

st.write('Prediction')
st.write(pred_df)

# Output result
name = pred_df.idxmax(axis=1).tolist()
st.write('Result')
st.write('The predicted Iris flower type is', name[0])