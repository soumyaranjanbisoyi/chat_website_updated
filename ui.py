import os
import streamlit as st
import pandas as pd
import json
import main  # Replace this with your actual module that contains the answer functions

# Set up your page configuration
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
from test2 import *
import constants
# Function to load the data from the JSON file
def load_session_data():
    try:
        with open("websites_session.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


# Function to save the data to the JSON file
def save_session_data(data):
    with open("websites_session.json", "w") as f:
        json.dump(data, f, indent=4)


# Sidebar: Website Management Section
st.sidebar.subheader("Website Management")
websites_data = load_session_data()

# Display current websites
st.sidebar.subheader("Current Websites:")
for site in websites_data:
    st.sidebar.text(site)

# Remove a website
remove_site = st.sidebar.selectbox("Remove a Website", websites_data)
if st.sidebar.button("Remove"):
    if remove_site:
        websites_data.remove(remove_site)
        save_session_data(websites_data)
        st.sidebar.success(f"Website {remove_site} removed successfully!")

# Add a new website
new_site = st.sidebar.text_input("Add New Website URL", key="new_site")
if st.sidebar.button("Add Website"):
    if new_site and new_site not in websites_data:
        websites_data.append(new_site)
        #
        # #### New code
        site_name = new_site.split('//')[1].split('/')[0]
        text_content = get_text_content(new_site)
        file_name = save_to_txt(site_name, text_content)
        store_in_vector(file_name, "vector_db")
        with open("config.json", "r") as file:
            vector_db_mapper = json.load(file)
        vector_db_mapper[new_site] = os.path.join("vector_db", file_name)
        with open("config.json", "w") as file2:
            json.dump(vector_db_mapper, file2)
        # #### New code
        save_session_data(websites_data)
        st.sidebar.success(f"Website {new_site} added successfully!")

col1, col2, col3 = st.columns((3, 3, 1))
# Main area for Q&A interface
with col2:
    st.title("Web Query AI")
col4, col5, col6 = st.columns((2.75, 3, 1))

with col5:
    st.subheader("A real time research solution")

# Input for user question
prompt = st.text_input("Enter your query here:", key="query")

# Button for generating individual answers
if st.button("Generate Individual Answers"):
    individual_answers = main.get_individual_answer(prompt)  # This function needs to exist in the 'main' module
    # Interchange the columns by changing the order of column names
    df_individual = pd.DataFrame(individual_answers.items(), columns=['Source', 'Response'])
    filtered_df = df_individual[~df_individual['Response'].str.contains('do not know', case=False, na=False)]
    filtered_df = filtered_df[~filtered_df['Response'].str.contains('can not determine', case=False, na=False)]
    st.table(filtered_df)  # Using Streamlit's built-in function to display the DataFrame as a table

# Button for generating a summarized answer
if st.button("Generate Summarized Answer"):
    summarized_answer = main.get_summarized_answer(prompt)  # This function also needs to exist in the 'main' module
    st.write("Summarized Answer:", summarized_answer)

# Ensure that your 'main' module has 'get_individual_answer' and 'get_summarized_answer' functions defined.
