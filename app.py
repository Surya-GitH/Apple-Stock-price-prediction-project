#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.title("Apple Stock Forecasting (SARIMA)")

# 1. Load the saved SARIMA model
@st.cache_resource
def load_model():
    return joblib.load("apple_sarima_model.pkl")

model_fit = load_model()

# 2. User Input: Choose how many steps to forecast
forecast_steps = st.slider("Select Forecast Horizon (Days)", min_value=1, max_value=30, value=5)

# 3. Generate Predictions
if st.button("Generate Forecast"):
    # Forecast future steps using the fitted SARIMA model
    forecast_result = model_fit.forecast(steps=forecast_steps)
    
    # Display results in a table or chart
    forecast_df = pd.DataFrame({
        "Day Ahead": range(1, forecast_steps + 1),
        "Predicted Close Price": np.asarray(forecast_result)
    })
    
    st.subheader("Forecast Results")
    st.dataframe(forecast_df)
    st.line_chart(forecast_df.set_index("Day Ahead"))

