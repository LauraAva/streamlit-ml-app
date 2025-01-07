import streamlit as st
import requests


# Title
st.title("👫Personas by CO2 Emission Class")

# Add another header for the next class
st.header("Low Emission Class (Class 1-2)")

# Create three columns for personas
col1, col2, col3 = st.columns(3)

## First Persona
with col1:
    st.image ("persona 1 lowemission 1-2 Europe.jpg", caption ="Western Europe Persona")
    st.write("**Demographics:** Urban dwellers, predominantly Middle-Class, with a significant proportion in the 15–29 years age range.")
    st.markdown("<div style='min-height: 100px;'></div>", unsafe_allow_html=True)
    st.write("**Insight:** This group is eco-conscious and likely benefits from advanced infrastructure supporting public transportation and renewable energy initiatives.")

# Second Persona
with col2:
    st.image("persona 2 lowemission 1-2 USA.jpg", caption="North America Persona")
    st.write("**Demographics:** City dwellers, mostly Upper-Class, with noticeable participation from the 30–49 years age range.")  
    st.markdown("<div style='min-height: 100px;'></div>", unsafe_allow_html=True)
    st.write("**Insight:** Luxury consumption patterns are mitigated by access to advanced green technologies.")

# Third Persona
with col3:
    st.image("persona 3 lowemission 1-2 Asia.jpg", caption="Asia Persona")
    st.write("**Demographics:** Small-town and rural populations with Middle-Class representation, primarily in the 30–49 years age range.")  
    st.markdown("<div style='min-height: 100px;'></div>", unsafe_allow_html=True)
    st.write("**Insight:** Investments in sustainable urban planning are showing results, though rural areas still lag.")

# Add a horizontal line for separation
st.markdown("---")

# Add another header for the next class
st.header("Medium Emission Class (Class 3-4)")

# Create three columns for the next personas
col4, col5, col6 = st.columns(3)

# First Persona
with col4:
    st.image("Persona 1 middleemission  Europe.jpg", caption="Western Europe Persona")
    st.write("**Demographics:** Suburban areas, Middle-Class dominance, with a tilt toward younger populations (15–29 years).")
    st.markdown("<div style='min-height: 100px;'></div>", unsafe_allow_html=True)
    st.write("**Insight:** Moderate emissions result from seasonal heating demands despite high environmental awareness.")

# Second Persona
with col5:
    st.image("Persona 1 middleemission USA.jpg", caption="North America Persona ")
    st.write("**Demographics:** Suburban populations, with a higher proportion of Middle-Class individuals and diverse age groups (15–49 years).")
    st.markdown("<div style='min-height: 100px;'></div>", unsafe_allow_html=True)
    st.write("**Insight:** Historical dependence on legacy systems for heating and energy contributes to moderate emission levels.")
    
# Third Persona
with col6:
    st.image("Persona 3 middleemission 3-4 Asia.jpg", caption="Asia Persona")
    st.write("**Demographics:** Predominantly Urban with representation from Middle-Class and Underprivileged groups; age distribution is skewed toward the 30–49 years bracket.")
    st.markdown("<div style='min-height: 100px;'></div>", unsafe_allow_html=True)
    st.write("**Insight:** Rapid industrialization and urban expansion contribute to emissions, despite government efforts for greener policies.")

# Add a horizontal line for separation
st.markdown("---")

# Add another header for the next class
st.header("High Emission Class (Class 5-6)")

# Create three columns for the next personas
col7, col8, col9 = st.columns(3)

# First Persona
with col7:
    st.image("Persona 1 highemission Europe.jpg", caption="Western Europe Persona")
    st.write("**Demographics:** Small-town residents, primarily Upper-Class individuals in the 30–49 years range.")
    st.markdown("<div style='min-height: 100px;'></div>", unsafe_allow_html=True)
    st.write("**Insight:** Consumption of luxury goods and reliance on imported energy sources contribute to this group’s high emissions.")

# Second Persona
with col8:
    st.image("Persona 2 highemission 5-6 USA.jpg", caption="North America Persona")
    st.write("**Demographics:** Rural Middle-Class and Upper-Class, with significant contributions from the 50–64 years age group.")
    st.markdown("<div style='min-height: 100px;'></div>", unsafe_allow_html=True)
    st.write("**Insight:** High per capita energy consumption in rural areas exacerbates emissions, driven by vehicle dependency and large residential footprints.")

# Third Persona
with col9:
    st.image("Persona 3 highemission Asia.jpg", caption="Asia Persona")
    st.write("**Demographics:** Rural and Urban Upper-Class with representation from younger to older (15–64 years).")
    st.markdown("<div style='min-height: 100px;'></div>", unsafe_allow_html=True)
    st.write("**Insight:** Industrial reliance and slower adaptation in rural areas contribute heavily to emissions.")
