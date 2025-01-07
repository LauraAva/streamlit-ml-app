import streamlit as st
import requests


# Title
st.title("👫Personas by CO2 Emission Class")

# Create three columns for personas
col1, col2, col3 = st.columns(3)

## First Persona
with col1:
    st.image ("persona 1 lowemission 1-2 Europe.jpg", caption ="Western Europe Persona - Low Emission Class Class 1-2")
    st.write("""
    ***Demographics:***
    Urban dwellers, predominantly Middle-Class, with a significant proportion in the 15–29 years age range.
    ***Insight:***
    This group is eco-conscious and likely benefits from advanced infrastructure supporting public transportation and renewable energy initiatives.
    """)

# Second Persona
with col2:
    st.image("persona 2 lowemission 1-2 USA.jpg", caption="North America Persona - Low Emission Class Class 1-2")
    st.write("""
    **Demographics:**  
    City dwellers, mostly Upper-Class, with noticeable participation from the 30–49 years age range.  
    **Insight:**  
    Luxury consumption patterns are mitigated by access to advanced green technologies.
    """)

# Third Persona
with col3:
    st.image("persona 3 lowemission 1-2 Asia.jpg", caption="Asia Persona - Low Emission Class Class 1-2")
    st.write("""
    **Demographics:**  
    Small-town and rural populations with Middle-Class representation, primarily in the 30–49 years age range.  
    **Insight:**  
    Investments in sustainable urban planning are showing results, though rural areas still lag.
    """)
# Add a horizontal line for separation
st.markdown("---")

# Add another header for the next class
st.header("Medium Emission Class (Class 3-4)")

# Create three columns for the next personas
col4, col5, col6 = st.columns(3)

# First Persona
with col4:
    st.image("path_to_image/persona4.png", caption="China Persona")
    st.write("""
    **Demographics:**  
    Predominantly Urban with representation from Middle-Class and Underprivileged groups; age distribution skewed toward the 30–49 years bracket.  
    **Insight:**  
    Rapid industrialization and urban expansion contribute to emissions, despite government efforts for greener policies.
    """)

# Second Persona
with col5:
    st.image("path_to_image/persona5.png", caption="Italy Persona")
    st.write("""
    **Demographics:**  
    Suburban populations, with a higher proportion of Middle-Class individuals and diverse age groups (15–49 years).  
    **Insight:**  
    Historical dependence on legacy systems for heating and energy contributes to moderate emission levels.
    """)

# Third Persona
with col6:
    st.image("path_to_image/persona6.png", caption="Sweden Persona")
    st.write("""
    **Demographics:**  
    Suburban areas, Middle-Class dominance, with a tilt toward younger populations (15–29 years).  
    **Insight:**  
    Moderate emissions result from seasonal heating demands despite high environmental awareness.
    """)
