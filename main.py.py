


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error



# In[26]:


# Function to load data
def load_data():
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.write("No file uploaded yet. Please upload a CSV file.")
        return None
    return data


# In[27]:


# Function to enter data manually
def manual_data_entry():
    st.write("Manual Data Entry")
    rows = st.number_input("Enter number of rows", min_value=1, step=1)
    cols = st.number_input("Enter number of columns", min_value=1, step=1)
    data = []
    for row in range(rows):
        data.append([st.text_input(f'Row {row+1} Col {col+1}', '') for col in range(cols)])
    if st.button("Submit"):
        data = pd.DataFrame(data)
        return data
    return None


# In[28]:


# Function to specify a data file path
def specify_file_path():
    st.write("Specify Data File Path")
    file_path = st.text_input("Enter file path")
    if file_path:
        data = pd.read_csv(file_path)
        return data
    return None



# In[29]:


# Function to display data
def display_data(data):
    st.write("Data Preview")
    st.dataframe(data.head())


# In[30]:


# Function to perform analysis
def perform_analysis(data):
    st.write("Performing Analysis")
    if data is not None:
        if st.checkbox("Show Raw Data"):
            st.write(data)
        
        target = st.selectbox("Select Target Column", data.columns)
        
        if target:
            X = data.drop(columns=[target])
            y = data[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            model = LinearRegression()
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            
            st.write("Mean Squared Error:", mse)

            st.write("Predictions vs Actual")
            comparison = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
            st.write(comparison.head())

# Main function
def main():
    st.title("Machine Learning Task")

    menu = ["Upload Data", "Specify Data File", "Enter Data Manually"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Upload Data":
        data = load_data()
    elif choice == "Specify Data File":
        data = specify_file_path()
    elif choice == "Enter Data Manually":
        data = manual_data_entry()
    else:
        data = None

    if data is not None:
        display_data(data)
        perform_analysis(data)

if __name__ == '__main__':
    main()






