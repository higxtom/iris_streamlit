import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

# Title
st.title("Streamlit Tutorial")
st.write("This is a simple Streamlit app.")

# DataFrames
st.write("## DataFrames")
df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
})

st.dataframe(df.style.highlight_max(axis=0), width=400, height=200)

st.table(df)

# 10 x 3 dataframes
st.write("## 10 x 3 dataframes")
chart_data = pd.DataFrame(
    np.random.randn(10, 3),
    columns=['a', 'b', 'c']
)
st.line_chart(chart_data)
st.area_chart(chart_data)
st.bar_chart(chart_data)

# Map
st.write("##  Map")
plot = pd.DataFrame(
    np.random.rand(100, 2) / [50, 50] + [35.69, 139.70],
    columns=['lat', 'lon']
)
st.map(plot)

# Image
img = Image.open("Iris.jpg")
st.image(img, caption="Iris", use_container_width=True)

# Checkbox
st.write("## Checkbox")
if st.checkbox('Show Image'):
    img = Image.open("Iris.jpg")
    st.image(img, caption="Iris", use_container_width=True)

# Slider
st.write("## Slider")
x = st.slider('x')
st.write(x, 'squared is', x * x)
