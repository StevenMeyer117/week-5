import streamlit as st

from apputil import *

# Load Titanic dataset
df = pd.read_csv(
    "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"
)

# Exercise 1
st.write(
    """
# Guiding Question (Exercise 1)
*How much more likely were you to survive if you were female as opposed to male?*
"""
)

# Exercise 2
st.write(
    """
# Exercise 2 – New Question
*How does family size affect average ticket fare within each passenger class?*  
This visualization explores whether larger families paid more, less, or similar fares 
compared to smaller groups or solo travelers.
"""
)

# Titanic Visualization 1
st.write(
    """
# Titanic Visualization 1  
Survival Rate by Passenger Class and Sex
"""
)
fig1 = visualize_demographic()
st.plotly_chart(fig1, use_container_width="stretch")

# Titanic Visualization 2
st.write(
    """
# Titanic Visualization 2  
Average Ticket Fare by Family Size and Passenger Class
"""
)
fig2 = visualize_families()
st.plotly_chart(fig2, use_container_width="stretch")