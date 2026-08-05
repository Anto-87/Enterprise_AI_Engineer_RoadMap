import streamlit as st 
import pandas as pd 
import numpy as np

# To run this app use streamlit run app.py

st.title("Hello Streamlit !!")

st.write("This is a simple text")

df = pd.DataFrame({
    'FirstColumn':[1,2,3,4],
    'SecondColumn': [10,20,30,40]
})


st.write("here is the dataframe")
st.write(df)

chart_data =pd.DataFrame(
    np.random.randn(20,3), columns=['a','b','c']
)
st.line_chart(chart_data)