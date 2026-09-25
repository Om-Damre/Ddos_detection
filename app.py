import streamlit as st
import pandas as pd
import joblib


# -------------------------------------------------
# Load trained model
# -------------------------------------------------

model = joblib.load("ddos_model.pkl")


# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="AI-Based DDoS Detection",
    page_icon="🛡️",
    layout="wide"
)


# -------------------------------------------------
# Title
# -------------------------------------------------

st.title("🛡️ AI-Based DDoS Attack Detection System")

st.write(
    "Enter network traffic parameters below to check "
    "whether the traffic is Normal or DDoS-like."
)

st.divider()


# -------------------------------------------------
# Input Section
# -------------------------------------------------

st.subheader("📊 Network Traffic Information")

col1, col2 = st.columns(2)

with col1:

    packets_per_second = st.number_input(
        "Packets per Second",
        min_value=0,
        max_value=20000,
        value=200
    )

    bytes_per_second = st.number_input(
        "Bytes per Second",
        min_value=0,
        max_value=2000000,
        value=20000
    )

    connection_count = st.number_input(
        "Connection Count",
        min_value=0,
        max_value=5000,
        value=10
    )


with col2:

    packet_size = st.number_input(
        "Average Packet Size",
        min_value=0,
        max_value=2000,
        value=800
    )

    failed_connections = st.number_input(
        "Failed Connections",
        min_value=0,
        max_value=1000,
        value=2
    )


st.divider()


# -------------------------------------------------
# Prediction Button
# -------------------------------------------------

if st.button("🔍 Analyze Network Traffic", use_container_width=True):

    # Create input dataframe
    input_data = pd.DataFrame([{
        "packets_per_second": packets_per_second,
        "bytes_per_second": bytes_per_second,
        "connection_count": connection_count,
        "packet_size": packet_size,
        "failed_connections": failed_connections
    }])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probability = model.predict_proba(input_data)[0][1] * 100


    # -------------------------------------------------
    # Result
    # -------------------------------------------------

    st.subheader("🔎 Detection Result")

    if prediction == 1:

        st.error("⚠️ DDoS ATTACK DETECTED")

        st.metric(
            "DDoS Probability",
            f"{probability:.2f}%"
        )

        st.warning(
            "The network traffic contains patterns "
            "similar to DDoS traffic."
        )

    else:

        st.success("✅ NORMAL TRAFFIC")

        st.metric(
            "DDoS Probability",
            f"{probability:.2f}%"
        )

        st.info(
            "The network traffic appears normal "
            "based on the trained model."
        )


    # -------------------------------------------------
    # Show Input Data
    # -------------------------------------------------

    st.subheader("📋 Analyzed Traffic")

    st.dataframe(
        input_data,
        use_container_width=True
    )


# -------------------------------------------------
# Footer
# -------------------------------------------------

st.divider()

st.caption(
    "AI-Based DDoS Attack Detection System | "
    "Machine Learning Demonstration"
)