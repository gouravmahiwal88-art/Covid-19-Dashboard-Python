# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: cpython-3.15.0b1-macos-aarch64-none (3.15.0)
#     language: python
#     name: python3
# ---

# %%
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as gr
from datetime import datetime, date
import os
from streamlit_folium import st_folium


# %%
# Set page config FIRST
st.set_page_config(
    page_title="COVID-19 Data Analytics & Forecasts",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# %%
# Core imports (after set_page_config)
from dashboard.data_loader import load_project_data
from dashboard.eda import get_kpis, get_top_affected_table, get_country_growth_rates
from dashboard.visualizations import (
    plot_cumulative_trends, plot_daily_changes, 
    plot_vaccination_donut, plot_correlation_heatmap, create_folium_map,
    THEME_COLORS, apply_dark_theme
)
from dashboard.chatbot import analyze_and_respond, inject_voice_assistant
from models.predictor import train_forecaster, analyze_death_rate_trends

# %%
# ==========================================
# 🎨 PREMIUM CSS GLASSMORPHISM DARK THEME
# ==========================================
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Main container and fonts */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Custom background gradient */
    .stApp {
        background: linear-gradient(135deg, #090b11 0%, #121824 100%);
        color: #f7fafc;
    }
    
    /* Custom styled Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(10, 12, 20, 0.9) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Glassmorphism KPI Card Styles */
    .kpi-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        transition: all 0.3s ease;
        margin-bottom: 15px;
    }
    
    .kpi-card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 242, 254, 0.3);
        box-shadow: 0 12px 40px 0 rgba(0, 242, 254, 0.15);
    }
    
    .kpi-title {
        font-size: 14px;
        color: #a0aec0;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.8px;
        margin-bottom: 6px;
    }
    
    .kpi-val {
        font-size: 32px;
        font-weight: 700;
        margin: 0;
        background: linear-gradient(to right, #ffffff, #e2e8f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .kpi-delta {
        font-size: 13px;
        font-weight: 600;
        margin-top: 6px;
    }
    
    /* Glowing Border Colors for specific cards */
    .border-teal {
        border-left: 5px solid #00d2ff !important;
    }
    .border-rose {
        border-left: 5px solid #ff3366 !important;
    }
    .border-green {
        border-left: 5px solid #2ecc71 !important;
    }
    .border-purple {
        border-left: 5px solid #9b59b6 !important;
    }
    
    /* Metric Delta Text Colors */
    .delta-up {
        color: #ff3366;
    }
    .delta-down {
        color: #2ecc71;
    }
    
    /* Clean margins for titles */
    h1, h2, h3 {
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Download Button overrides */
    div.stDownloadButton > button {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%) !important;
        border: none !important;
        color: #000000 !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        box-shadow: 0 4px 15px rgba(0, 242, 254, 0.3) !important;
        transition: all 0.2s ease !important;
    }
    
    div.stDownloadButton > button:hover {
        transform: scale(1.03) !important;
        box-shadow: 0 6px 20px rgba(0, 242, 254, 0.5) !important;
    }
    
    /* Chat bubbles styling */
    .chat-bubble-user {
        background: rgba(43, 108, 176, 0.4);
        border: 1px solid rgba(66, 153, 225, 0.4);
        border-radius: 12px 12px 0 12px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    .chat-bubble-bot {
        background: rgba(29, 36, 51, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px 12px 12px 0;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# %%
# ==========================================
# 🎛️ SIDEBAR CONTROL SYSTEM
# ==========================================
st.sidebar.markdown("<h2 style='margin-top:0;'>📊 Project Controls</h2>", unsafe_allow_html=True)

# %%
# A. Data Mode Option
data_mode = st.sidebar.selectbox(
    "Data Source / Environment",
    ["Offline (Cleaned)", "Offline (Raw Messy)", "Live Global API (disease.sh)"],
    help="Select 'Offline (Raw Messy)' to see simulated data before cleaning, or 'Live' to fetch live API statistics."
)

# %%
# B. Loading Datasets
df_cases, df_vax, load_success = load_project_data(data_mode)

# %%
if not load_success:
    if "Live" in data_mode:
        st.sidebar.error("⚠️ API fetch failed! Falling back to cached Offline datasets.")
    else:
        st.sidebar.error("⚠️ Error loading CSV files. Run data/generate_sample_data.py first.")

# %%
# Get chronological bounds
df_cases['Date'] = pd.to_datetime(df_cases['Date'])
min_date = df_cases['Date'].min().to_pydatetime()
max_date = df_cases['Date'].max().to_pydatetime()

# %%
# C. Country Selector
countries = sorted(df_cases['Country'].unique())
selected_country = st.sidebar.selectbox(
    "Focus Area (Country)",
    ["Global"] + list(countries),
    help="Select 'Global' for world aggregates or a specific country for details."
)

# %%
# D. Date Filter
st.sidebar.markdown("### 📅 Date Limits")
date_range = st.sidebar.date_input(
    "Select Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Filters the analytics timeline."
)

# %%
# E. Machine Learning Model Panel (Streamlined)
st.sidebar.markdown("### 🔮 ML Regression Panel")
forecast_horizon = st.sidebar.slider(
    "Projection Horizon (Days)",
    min_value=7, max_value=90, value=30, step=7,
    help="How many days into the future the Linear/Polynomial model will project cases."
)

# %%
poly_deg = st.sidebar.slider(
    "Polynomial Order (Degree)",
    min_value=1, max_value=3, value=2,
    help="1 = Standard Linear Regression, 2 = Quadratic Curves, 3 = Cubic Curves."
)

# %%
# Apply filters
if len(date_range) == 2:
    start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_cases = df_cases[(df_cases['Date'] >= start_dt) & (df_cases['Date'] <= end_dt)]
    if not df_vax.empty:
        df_vax['Date'] = pd.to_datetime(df_vax['Date'])
        filtered_vax = df_vax[(df_vax['Date'] >= start_dt) & (df_vax['Date'] <= end_dt)]
    else:
        filtered_vax = df_vax
else:
    filtered_cases = df_cases
    filtered_vax = df_vax

# %%
# ==========================================
# 📊 METRICS / KPI GRID
# ==========================================
kpis = get_kpis(filtered_cases, filtered_vax, selected_country)

# %%
st.markdown("<h1 style='margin-bottom:0;'>🦠 COVID-19 Data Analytics Dashboard</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:#a0aec0; margin-top:2px;'>Scope: <b>{selected_country}</b> | Source Mode: <b>{data_mode}</b></p>", unsafe_allow_html=True)

# %%
# Render KPIs using HTML for full visual control
kpi_cols = st.columns(4)

# %%
with kpi_cols[0]:
    st.markdown(f"""
    <div class="kpi-card border-teal">
        <div class="kpi-title">Confirmed Cases</div>
        <div class="kpi-val">{kpis['cases']:,}</div>
        <div class="kpi-delta"><span class="delta-up">▲ +{kpis['new_cases']:,}</span> (24h)</div>
    </div>
    """, unsafe_allow_html=True)

# %%
with kpi_cols[1]:
    st.markdown(f"""
    <div class="kpi-card border-rose">
        <div class="kpi-title">Total Fatalities</div>
        <div class="kpi-val">{kpis['deaths']:,}</div>
        <div class="kpi-delta"><span class="delta-up">▲ +{kpis['new_deaths']:,}</span> (24h)</div>
    </div>
    """, unsafe_allow_html=True)

# %%
with kpi_cols[2]:
    st.markdown(f"""
    <div class="kpi-card border-green">
        <div class="kpi-title">Recovered Patients</div>
        <div class="kpi-val">{kpis['recovered']:,}</div>
        <div class="kpi-delta" style="color:#2ecc71;">🏥 {kpis['recovery_rate']:.2f}% Rate</div>
    </div>
    """, unsafe_allow_html=True)

# %%
with kpi_cols[3]:
    vax_details = f"💉 {kpis['vax_rate']:.2f}% immunized" if kpis['doses'] > 0 else "N/A"
    doses_str = f"{kpis['doses']:,}" if kpis['doses'] > 0 else "No Record"
    st.markdown(f"""
    <div class="kpi-card border-purple">
        <div class="kpi-title">Vaccines Administered</div>
        <div class="kpi-val">{doses_str}</div>
        <div class="kpi-delta" style="color:#9b59b6;">{vax_details}</div>
    </div>
    """, unsafe_allow_html=True)

# %%
# ==========================================
# 🗂️ MAIN TABS INTERFACE
# ==========================================
tab_realtime, tab_cleaning, tab_ml, tab_vax, tab_chatbot = st.tabs([
    "📊 Real-Time Analytics", 
    "🧼 Data Cleaning Pipeline", 
    "🔮 ML Predictions & Forecasts", 
    "💉 Immunization Progress",
    "💬 Voice & Text AI Chatbot"
])

# %%
# ------------------------------------------
# TAB 1: REAL-TIME ANALYTICS
# ------------------------------------------
with tab_realtime:
    st.subheader("📈 Epidemiological Trends & Geospatial Overlays")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Determine focal country
        chart_country = "United States" if selected_country == "Global" else selected_country
        st.plotly_chart(plot_cumulative_trends(filtered_cases, chart_country), width="stretch")
        st.plotly_chart(plot_daily_changes(filtered_cases, chart_country), width="stretch")
        
    with col2:
        st.markdown("<h4 style='margin-bottom:12px;'>🌍 Global Infection Geospatial Density</h4>", unsafe_allow_html=True)
        # Render Folium Map using st_folium
        try:
            m = create_folium_map(filtered_cases, filtered_vax)
            st_folium(m, width=700, height=450)
        except Exception as e:
            st.error(f"Geomap rendering error: {e}")
            st.info("Check internet connectivity for Folium background tiles.")
            
        st.markdown("#### 🏆 Global Rankings Table")
        ranking_df = get_top_affected_table(filtered_cases, filtered_vax)
        st.dataframe(ranking_df.head(10), hide_index=True)
        
        # Download Table CSV Button
        csv_data = ranking_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Global COVID-19 Report (CSV)",
            data=csv_data,
            file_name=f"COVID19_Global_Report_{date.today()}.csv",
            mime="text/csv"
        )

# %%
# ------------------------------------------
# TAB 2: DATA CLEANING PIPELINE
# ------------------------------------------
with tab_cleaning:
    st.subheader("🧼 In-Memory Automated Data Cleaning Pipeline")
    st.markdown("""
    This section showcases the raw, messy dataset simulating actual uncleaned imports (which you can load via the **Data Source** sidebar control) versus the fully structured, cleaned analytical pipeline.
    """)
    
    col_raw, col_clean = st.columns(2)
    
    with col_raw:
        st.markdown("<h4 style='color:#ff3366;'>❌ Messy Raw Ingestion State</h4>", unsafe_allow_html=True)
        raw_c = pd.read_csv('data/covid_data_raw.csv')
        raw_v = pd.read_csv('data/vaccination_data_raw.csv')
        
        st.markdown(f"**Raw Cases Rows**: {len(raw_c)} | **Raw Vax Rows**: {len(raw_v)}")
        st.dataframe(raw_c.head(120), height=300)
        
        st.markdown("""
        > **Identified Raw Issues:**
        > 1. **Mixed Date Formats**: Dates are written in various layouts (e.g. `YYYY-MM-DD`, `DD/MM/YYYY`, `MM-DD-YYYY` randomly).
        > 2. **Duplicate Records**: Multiple redundant entries for identical dates and country targets are injected.
        > 3. **Null Values (NaN)**: Crucial cumulative metrics contain missing fields, leading to numerical errors if left unhandled.
        """)
        
    with col_clean:
        st.markdown("<h4 style='color:#00d2ff;'>✅ Analytical Cleaned State</h4>", unsafe_allow_html=True)
        cleaned_c, cleaned_v = load_project_data("Offline (Cleaned)")[:2]
        
        st.markdown(f"**Cleaned Cases Rows**: {len(cleaned_c)} (Deduplicated) | **Cleaned Vax Rows**: {len(cleaned_v)}")
        st.dataframe(cleaned_c.head(120), height=300)
        
        # Calculate clean differentials
        dup_removed_cases = len(raw_c) - len(cleaned_c)
        dup_removed_vax = len(raw_v) - len(cleaned_v)
        
        st.markdown(f"""
        > **Executed In-Memory Pipeline Actions:**
        > - **Standardized Date Formats**: Automatically parsed all dates into consistent `YYYY-MM-DD` ISO Datetime format.
        > - **Deduplicated Records**: Dropped **{dup_removed_cases}** redundant rows in cases and **{dup_removed_vax}** in vaccinations, preserving the latest updates.
        > - **Missing Values Rectified**: Addressed NaN entries by grouping by country and performing an automated **Forward-Fill (`ffill`) followed by a Backward-Fill (`bfill`)** pipeline.
        """)

# %%
# ------------------------------------------
# TAB 3: ML PREDICTIONS & FORECASTS
# ------------------------------------------
with tab_ml:
    st.subheader("🔮 Predictive Machine Learning Trend Modeling")
    st.markdown("""
    This tab trains a **Scikit-learn Polynomial Linear Regression model** dynamically on the filtered timeline to predict cumulative cases or deaths.
    Select a country in the sidebar and configure modeling options to see predictions and confidence bounds.
    """)
    
    ml_country = "United States" if selected_country == "Global" else selected_country
    
    col_opt1, col_opt2 = st.columns([1, 3])
    
    with col_opt1:
        st.markdown("##### ⚙️ Model Target")
        ml_target = st.radio("Forecast Indicator", ["Confirmed", "Deaths"])
        
        # Compute recent trend stats
        country_df = filtered_cases[filtered_cases['Country'] == ml_country].sort_values(by='Date')
        
        if len(country_df) > 10:
            latest_val = int(country_df[ml_target].iloc[-1])
            prev_val = int(country_df[ml_target].iloc[-7]) if len(country_df) >= 7 else int(country_df[ml_target].iloc[0])
            growth_7d = latest_val - prev_val
            
            st.markdown(f"""
            <div style="background:rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; margin-top: 15px; border: 1px solid rgba(255,255,255,0.1);">
                <span style="font-size:12px; color:#a0aec0;">Latest Value ({ml_country})</span><br>
                <b style="font-size:20px;">{latest_val:,}</b><br>
                <span style="font-size:12px; color:#a0aec0;">7-Day Increase</span><br>
                <b style="font-size:16px; color:#ff3366;">+{growth_7d:,}</b>
            </div>
            """, unsafe_allow_html=True)
            
    with col_opt2:
        # Train and forecast
        forecast_res = train_forecaster(country_df, target_col=ml_target, forecast_days=forecast_horizon, poly_degree=poly_deg)
        
        if forecast_res:
            df_fit = forecast_res['df_fit']
            df_forecast = forecast_res['df_forecast']
            metrics = forecast_res['metrics']
            
            # Plot ML Chart
            fig_ml = gr.Figure()
            
            # 1. Historical Actual Points
            fig_ml.add_trace(gr.Scatter(
                x=df_fit['Date'], y=df_fit['Actual'],
                mode='markers', name='Actual Data',
                marker=dict(color='rgba(255, 255, 255, 0.4)', size=4),
                hovertemplate='%{y:,}'
            ))
            
            # 2. Historical Fitted Curve
            fig_ml.add_trace(gr.Scatter(
                x=df_fit['Date'], y=df_fit['Fitted'],
                mode='lines', name='Model Curve Fit',
                line=dict(color=THEME_COLORS['confirmed'], width=2),
                hovertemplate='%{y:,.0f}'
            ))
            
            # 3. Future Projected Line
            fig_ml.add_trace(gr.Scatter(
                x=df_forecast['Date'], y=df_forecast['Forecast'],
                mode='lines', name='ML Projection Horizon',
                line=dict(color=THEME_COLORS['vaccinated'], width=2.5, dash='dash'),
                hovertemplate='%{y:,.0f}'
            ))
            
            # 4. Confidence Interval Bounds (Shaded Area)
            fig_ml.add_trace(gr.Scatter(
                x=pd.concat([df_forecast['Date'], df_forecast['Date'][::-1]]),
                y=pd.concat([df_forecast['Upper_Bound'], df_forecast['Lower_Bound'][::-1]]),
                fill='toself',
                fillcolor='rgba(155, 89, 182, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name='95% Prediction Interval'
            ))
            
            fig_ml.update_layout(
                title=f"{ml_country} - {ml_target} Forecast (Polynomial Degree {poly_deg})",
                xaxis_title="Date",
                yaxis_title="Total Counts"
            )
            
            st.plotly_chart(apply_dark_theme(fig_ml), width="stretch")
            
            # Show Metrics Grid
            met_cols = st.columns(4)
            with met_cols[0]:
                st.metric("Coefficient of Determination (R²)", f"{metrics['r2']:.4f}")
            with met_cols[1]:
                st.metric("Root Mean Squared Error (RMSE)", f"{int(metrics['rmse']):,}")
            with met_cols[2]:
                st.metric("Current Value", f"{int(df_fit['Actual'].iloc[-1]):,}")
            with met_cols[3]:
                st.metric(f"Projected Value (+{forecast_horizon} days)", f"{int(df_forecast['Forecast'].iloc[-1]):,}")
                
        else:
            st.warning("⚠️ Could not compile regression model. Verify that you have chosen a country with adequate statistics.")

# %%
# ------------------------------------------
# TAB 4: IMMUNIZATION PROGRESS
# ------------------------------------------
with tab_vax:
    st.subheader("💉 Vaccination Penetration & Immunity Status")
    
    col_vax1, col_vax2 = st.columns([3, 2])
    
    with col_vax1:
        vax_country = "United States" if selected_country == "Global" else selected_country
        latest_c_vax = filtered_cases[filtered_cases['Country'] == vax_country].sort_values(by='Date')
        
        if not latest_c_vax.empty:
            pop_val = int(latest_c_vax['Population'].iloc[-1])
            st.plotly_chart(plot_vaccination_donut(filtered_vax, vax_country, pop_val), width="stretch")
        else:
            st.info("No population record found to assess vaccine ratio bounds.")
            
    with col_vax2:
        st.markdown("#### 📊 Statistical Cross-Indicator Correlations")
        st.markdown("Analyzes mathematical linkages between confirmed cases, densities, death metrics, and total immunizations globally.")
        
        st.plotly_chart(plot_correlation_heatmap(filtered_cases, filtered_vax), width="stretch")

# %%
# ------------------------------------------
# TAB 5: VOICE & TEXT AI CHATBOT
# ------------------------------------------
with tab_chatbot:
    st.subheader("💬 AI Conversational Analytics Assistant")
    st.markdown("""
    Ask questions about COVID-19 cases, deaths, recoveries, vaccinations, and forecasts in natural language.
    Click the **Microphone** button to speak, or type your query in the input below. Click the **Speaker** button to hear the bot's response!
    """)
    
    # Simple Session State to store chat conversations
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = [
            ("bot", "👋 Hello! I am your COVID-19 Data Assistant. Speak into your microphone or type a question about infection statistics or machine learning predictions to get started!")
        ]
        
    # Helper to send message
    def send_chat_message(usr_msg):
        if usr_msg and usr_msg.strip() != "":
            # Add user message
            st.session_state['chat_history'].append(("user", usr_msg))
            # Generate bot response
            bot_res = analyze_and_respond(usr_msg, filtered_cases, filtered_vax)
            st.session_state['chat_history'].append(("bot", bot_res))
            # Rerun
            st.rerun()
            
    # Quick prompt buttons
    st.markdown("💡 *Quick Queries:*")
    qp_cols = st.columns(4)
    with qp_cols[0]:
        if st.button("🇺🇸 USA cases overview?", width="stretch"):
            send_chat_message("What are the total cases and new cases in USA?")
    with qp_cols[1]:
        if st.button("🇮🇳 India vaccination status?", width="stretch"):
            send_chat_message("What is the vaccination rate in India?")
    with qp_cols[2]:
        if st.button("🔮 Predict cases for Brazil?", width="stretch"):
            send_chat_message("Predict confirmed cases for Brazil next 30 days")
    with qp_cols[3]:
        if st.button("🌍 Highest death rate?", width="stretch"):
            send_chat_message("Which country has the highest death rate?")

    # Display scrollable conversation history
    st.markdown("---")
    chat_box = st.container()
    
    with chat_box:
        for sender, msg in st.session_state['chat_history']:
            if sender == "user":
                st.markdown(f'<div class="chat-bubble-user">👤 <b>You:</b><br>{msg}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bubble-bot">🤖 <b>AI Assistant:</b><br>{msg}</div>', unsafe_allow_html=True)
                
    st.markdown("---")
    
    # Input panel
    chat_input_val = st.text_input(
        "Type your question:", 
        key="chat_text_box",
        placeholder="Ask about COVID-19 data...",
        label_visibility="collapsed"
    )
    
    sub_col1, sub_col2 = st.columns([6, 1])
    with sub_col1:
        if st.button("💬 Send Message", width="stretch"):
            if chat_input_val:
                send_chat_message(chat_input_val)
    with sub_col2:
        if st.button("🧹 Clear Chat", width="stretch"):
            st.session_state['chat_history'] = [
                ("bot", "👋 Conversation cleared! Ask me anything about COVID-19 data.")
            ]
            st.rerun()
            
    # Inject Voice Synthesizer / Web Speech interface for the LAST bot response
    latest_bot_response = ""
    for sender, msg in reversed(st.session_state['chat_history']):
        if sender == "bot":
            latest_bot_response = msg
            break
            
    if latest_bot_response:
        inject_voice_assistant(latest_bot_response)

# %%
# ==========================================
# 📑 FOOTER DESIGN
# ==========================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #a0aec0; font-size: 12px; margin-top:20px;'>"
    "COVID-19 Data Analytics & Machine Learning Project. Created using Python, Streamlit, Plotly, Scikit-learn, and Folium. "
    "Features offline messy data pipelines and live disease.sh real-time integrations."
    "</p>",
    unsafe_allow_html=True
)

