import streamlit as st
import requests


# Title
st.title("👫Personas by CO2 Emission Class")

# Add another header for the next class
st.header("Low Emission Class (Class 1-2)")

# Define CSS for consistent alignment
st.markdown("""
    <style>
    .persona-container {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 250px; /* Adjust height to match the longest content */
    }
    </style>
""", unsafe_allow_html=True)

# Create three columns for personas
col1, col2, col3 = st.columns(3)

## First Persona
with col1:
    st.image ("persona 1 lowemission 1-2 Europe.jpg", caption ="Western Europe Persona")
    st.markdown(
        """
        <div class="persona-container">
            <p><strong>Demographics:</strong> Urban dwellers, predominantly Middle-Class, with a significant proportion in the 15–29 years age range.</p>
            <p><strong>Insight:</strong> This group is eco-conscious and likely benefits from advanced infrastructure supporting public transportation and renewable energy initiatives.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Second Persona
with col2:
    st.image("persona 2 lowemission 1-2 USA.jpg", caption="North America Persona")
    st.markdown(
        """
        <div class="persona-container">
            <p><strong>Demographics:</strong> City dwellers, mostly Upper-Class, with noticeable participation from the 30–49 years age range.</p>
            <p><strong>Insight:</strong> Luxury consumption patterns are mitigated by access to advanced green technologies.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
# Third Persona
with col3:
    st.image("persona 3 lowemission 1-2 Asia.jpg", caption="Asia Persona")
    st.markdown(
        """
        <div class="persona-container">
            <p><strong>Demographics:</strong> Small-town and rural populations with Middle-Class representation, primarily in the 30–49 years age range.</p>
            <p><strong>Insight:</strong> Investments in sustainable urban planning are showing results, though rural areas still lag.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
# Add a horizontal line for separation
st.markdown("---")

# Add another header for the next class
st.header("Medium Emission Class (Class 3-4)")

# Create three columns for the next personas
col4, col5, col6 = st.columns(3)

# First Persona
with col4:
    st.image("Persona 1 middleemission  Europe.jpg", caption="Western Europe Persona")
    st.markdown(
        """
        <div class="persona-container">
            <p><strong>Demographics:</strong> Suburban areas, Middle-Class dominance, with a tilt toward younger populations (15–29 years).</p>
            <p><strong>Insight:</strong> Moderate emissions result from seasonal heating demands despite high environmental awareness.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
# Second Persona
with col5:
    st.image("Persona 1 middleemission USA.jpg", caption="North America Persona ")
    st.markdown(
        """
        <div class="persona-container">
            <p><strong>Demographics:</strong> Suburban populations, with a higher proportion of Middle-Class individuals and diverse age groups (15–49 years).</p>
            <p><strong>Insight:</strong> Historical dependence on legacy systems for heating and energy contributes to moderate emission levels.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
# Third Persona
with col6:
    st.image("Persona 3 middleemission 3-4 Asia.jpg", caption="Asia Persona")
    st.markdown(
        """
        <div class="persona-container">
            <p><strong>Demographics:</strong> Predominantly Urban with representation from Middle-Class and Underprivileged groups; age distribution is skewed toward the 30–49 years bracket.</p>
            <p><strong>Insight:</strong> Rapid industrialization and urban expansion contribute to emissions, despite government efforts for greener policies.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
# Add a horizontal line for separation
st.markdown("---")

# Add another header for the next class
st.header("High Emission Class (Class 5-6)")

# Create three columns for the next personas
col7, col8, col9 = st.columns(3)

# First Persona
with col7:
    st.image("Persona 1 highemission Europe.jpg", caption="Western Europe Persona")
    st.markdown(
        """
        <div class="persona-container">
            <p><strong>Demographics:</strong> Small-town residents, primarily Upper-Class individuals in the 30–49 years range.</p>
            <p><strong>Insight:</strong> Consumption of luxury goods and reliance on imported energy sources contribute to this group’s high emissions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Second Persona
with col8:
    st.image("Persona 2 highemission 5-6 USA.jpg", caption="North America Persona")
    st.markdown(
        """
        <div class="persona-container">
            <p><strong>Demographics:</strong> Rural Middle-Class and Upper-Class, with significant contributions from the 50–64 years age group.</p>
            <p><strong>Insight:</strong> High per capita energy consumption in rural areas exacerbates emissions, driven by vehicle dependency and large residential footprints.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
# Third Persona
with col9:
    st.image("Persona 3 highemission Asia.jpg", caption="Asia Persona")
    st.markdown(
        """
        <div class="persona-container">
            <p><strong>Demographics:</strong> Rural and Urban Upper-Class with representation from younger to older (15–64 years).</p>
            <p><strong>Insight:</strong> Industrial reliance and slower adaptation in rural areas contribute heavily to emissions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
# Add a horizontal line for separation
st.markdown("---")
