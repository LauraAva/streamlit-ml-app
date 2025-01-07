import streamlit as st
import requests

# Set up the sidebar
setup_sidebar()

# Title
st.title("👫Personas by CO2 Emission Class")

# Load descriptions
description_url = "https://raw.githubusercontent.com/your-username/streamlit-persona-app/main/persona_descriptions.md"
response = requests.get(description_url)
if response.status_code == 200:
    descriptions = response.text.split("\n\n")  # Assuming each persona is separated by a double newline
else:
    st.error("Failed to load descriptions.")

# Persona Images and Descriptions
persona_images = [
    "https://raw.githubusercontent.com/your-username/streamlit-persona-app/main/persona1.png",
    "https://raw.githubusercontent.com/your-username/streamlit-persona-app/main/persona2.png",
    # Add more URLs for additional personas
]

for i, img_url in enumerate(persona_images):
    st.image(img_url, caption=f"Persona {i + 1}", use_column_width=True)
    if i < len(descriptions):
        st.markdown(descriptions[i])
