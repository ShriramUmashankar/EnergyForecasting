import streamlit as st
import streamlit.components.v1 as components
import requests
import time

# [AI GENERATED]

# ==========================================
# CONFIG & SETUP
# ==========================================
API_URL = "http://fastapi-backend:8000"

st.set_page_config(
    page_title="Energy Forecast Dashboard",
    page_icon="⚡",
    layout="wide"
)

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def fetch_metrics():
    try:
        response = requests.get(f"{API_URL}/stats")
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.ConnectionError:
        st.error("🚨 Cannot connect to the backend. Is FastAPI running on port 8000?")
    return None

def fetch_plot(endpoint):
    try:
        response = requests.get(f"{API_URL}{endpoint}")
        if response.status_code == 200:
            return response.text
    except Exception:
        pass
    return None

# ==========================================
# UI LAYOUT
# ==========================================
st.title("⚡ Energy Consumption Forecaster")

# Create tabs for clean UX
tab_live, tab_upload, tab_manual = st.tabs([
    "🔴 Live Monitoring & Weekly Forecast", 
    "📁 Custom Batch Upload", 
    "📖 User Manual"
])

# ------------------------------------------
# TAB 1: LIVE MONITORING
# ------------------------------------------
with tab_live:
    st.markdown("### Real-Time Monitoring & Rolling Forecast")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh data every 5 seconds", value=False)
    
    metrics_data = fetch_metrics()
    
    if metrics_data:
        # Display top-level metrics clearly
        col1, col2, col3 = st.columns(3)
        col1.metric("Samples Processed", metrics_data.get("samples", 0))
        col2.metric("Overall RMSE", f"{metrics_data.get('rmse', 0):.4f}")
        col3.metric("Overall MAE", f"{metrics_data.get('mae', 0):.4f}")
        
        st.divider() # Adds a clean visual separator
        
        # Fetch both plots from the FastAPI backend
        live_html = fetch_plot("/plot/live")
        forecast_html = fetch_plot("/plot/week-forecast")
        
        # Render Live Hourly Plot
        st.subheader("Live Next-Hour Prediction vs Actual")
        if live_html and "No prediction data yet" not in live_html:
            components.html(live_html, height=450, scrolling=False)
        else:
            st.info("Waiting for your external script to send prediction data...")
            
        st.write("") # Add a little vertical spacing
        
        # Render Rolling 7-Day Forecast Plot
        st.subheader("Rolling 168-Hour Forecast")
        if forecast_html and "No prediction data yet" not in forecast_html:
            components.html(forecast_html, height=450, scrolling=False)
        else:
            st.info("Waiting for initial data to generate the 7-day forecast...")
            
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(5)
        st.rerun()

# ------------------------------------------
# TAB 2: BATCH UPLOAD (7-DAY FORECAST)
# ------------------------------------------
with tab_upload:
    st.markdown("### Generate a 7-Day Forecast")
    st.markdown("Upload a CSV file containing at 25 hours of data (Current hour metrics and previous 24 hours data) to generate a 168-hour forecast.")
    
    # Foolproof UX: Restrict file type
    uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])
    
    if uploaded_file is not None:
        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Processing data and generating recursive forecast..."):
                try:
                    # Post file to FastAPI
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    res = requests.post(f"{API_URL}/upload", files=files)
                    
                    if res.status_code == 200:
                        data = res.json()
                        if "error" in data:
                            st.warning(f"⚠️ {data['error']}")
                        else:
                            st.success("✅ Forecast generated successfully!")
                            
                            # Fetch and display the forecast plot
                            forecast_html = fetch_plot("/plot/week-forecast")
                            if forecast_html:
                                components.html(forecast_html, height=500, scrolling=False)
                    else:
                        st.error(f"Failed to process file. Status Code: {res.status_code}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

# ------------------------------------------
# TAB 3: USER MANUAL
# ------------------------------------------
with tab_manual:
    st.markdown("""
    ### 📖 User Manual & Troubleshooting
    
    **Overview**
    This application predicts energy consumption (Global Active Power) in real-time and provides long-term 7-day forecasts.
    
    **The Live Monitor:**
    1. The live monitor shows the predicted model output for the next hour along with previous predicted hours.
    2. Check the **"Auto-refresh"** box in the Live Monitoring tab.
    3. The application will automatically update the RMSE/MAE metrics and plot the predicted vs. actual values. Hover your mouse over any point on the graph to see the exact values.
    4. The second plot shows weekly energy consumption prediction.
    
    **How to use Custom Batch Upload:**
    1. Navigate to the **Batch Upload** tab.
    2. Drag and drop a valid `.csv` file. The file **must** contain the following columns: `timestamp`, `Global_active_power`, `Global_reactive_power`, `Voltage`, `Global_intensity`, `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`.
    3. The file must have a minimum of 25 rows to fulfill the historical data buffer.
    4. Click "Generate Forecast" and wait a few seconds. It will show the weekly prediction on this custom batch.
    
    **Understanding Metrics:**
    * **RMSE (Root Mean Square Error):** Measures the average magnitude of the error. Lower is better.
    * **MAE (Mean Absolute Error):** The average absolute difference between predictions and actuals. Lower is better.
    """)