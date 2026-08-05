import streamlit as st
import pandas as pd

st.title("Streamlit Text Input")

# Input 
name= st.text_input("Enter your name:")

# Slider
age =st.slider("Select your age:",0,100,25)

if name:
    st.write(f"Hello, {name}")
    
st.write(f"Your age is {age}")

# SelectItem
options=["Python", "Java","C++", "JavaScript"]
choice = st.selectbox("choose your favourite language:", options)

st.write(f"Your selected {choice}")


# Dataframe
data = {
    "Name":["John","Jake","Smith", "Jill"],
    "Age":[28,24,32,21],
    "City":["New York","Los Angeles","Chicago","Houston"]
}

df = pd.DataFrame(data)
df.to_csv("sampledata.csv")
st.write(df)


# File uploader
uploaded_file =st.file_uploader("Choose a CSV file",type="csv")

if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    st.write(df)
    