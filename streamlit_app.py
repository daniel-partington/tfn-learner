import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc, gammaln

st.set_page_config(page_title="TFN Teaching Tool", layout="wide")

# --- Helper Functions ---
def gamma_irf(t, A, n, a):
    """Continuous Gamma Impulse Response Function."""
    t = np.asarray(t, dtype=float)
    return A * (t ** (n - 1)) * np.exp(-t / a) / (a**n * np.exp(gammaln(n)))

def gamma_block(t, A, n, a, dt=1):
    """
    Gamma Block Response: Integrated IRF over time step dt.
    Using the regularized lower incomplete gamma function.
    """
    t = np.asarray(t, dtype=float)
    lower = np.clip(t - dt, a_min=0.0, a_max=None)
    return A * (gammainc(n, t / a) - gammainc(n, lower / a))

# --- Sidebar Navigation ---
st.sidebar.title("TFN Academy")
page = st.sidebar.radio("Select a Lesson", ["1. The Impulse (IRF)", "2. The Block Response", "3. Convolution & Head"])

# --- Page 1: The IRF ---
if page == "1. The Impulse (IRF)":
    st.header("Step 1: The Impulse Response Function (IRF)")
    st.write("The IRF represents the aquifer's 'DNA'. It shows how the water level would react to a single, instantaneous pulse of rain.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Aquifer Parameters")
        A = st.slider("Gain (A): Total Rise", 1.0, 10.0, 5.0)
        n = st.slider("Shape (n): Peakiness", 1.1, 5.0, 1.2)
        a = st.slider("Scale (a): Response Time", 10, 200, 100)
    
    with col2:
        t = np.linspace(0.1, 500, 500)
        y = gamma_irf(t, A, n, a)
        
        fig, ax = plt.subplots()
        ax.plot(t, y, color='blue', lw=2)
        ax.set_xlabel("Days since rain")
        ax.set_ylabel("Response Magnitude")
        ax.set_title("Theoretical Gamma IRF")
        st.pyplot(fig)

# --- Page 2: The Block Response ---
elif page == "2. The Block Response":
    st.header("Step 2: From Impulse to Block")
    st.write("Since we measure rain in daily 'blocks', we integrate the IRF over a 1-day window.")
    
    A = st.sidebar.slider("Gain (A)", 1.0, 10.0, 5.0)
    n = st.sidebar.slider("Shape (n)", 1.1, 5.0, 1.2)
    a = st.sidebar.slider("Scale (a)", 10, 200, 100)
    
    t = np.arange(1, 500)
    y_irf = gamma_irf(t, A, n, a)
    y_block = gamma_block(t, A, n, a, dt=1)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, y_irf, label="Continuous IRF", alpha=0.5, linestyle='--')
    ax.step(t, y_block, where='post', label="Daily Block Response", color='red')
    ax.set_title("How the model 'sees' 1mm of daily rain")
    ax.legend()
    st.pyplot(fig)

# --- Page 3: Convolution ---
elif page == "3. Convolution & Head":
    st.header("Step 3: Convolution (Summing it up)")
    st.write("Watch how multiple rain pulses accumulate over time to create the Groundwater Head.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        num_pulses = st.number_input("Number of rain pulses", 1, 10, 3)
        h0 = st.slider("Initial Head (Baseline)", 10.0, 50.0, 20.0)
        
        pulses = []
        for i in range(num_pulses):
            p_time = st.slider(f"Day of Pulse {i+1}", 0, 100, i*20)
            p_amt = st.slider(f"Amount (mm) {i+1}", 1, 20, 10)
            pulses.append((p_time, p_amt))
            
        A = st.sidebar.slider("Gain (A)", 1.0, 10.0, 2.0)
        n = st.sidebar.slider("Shape (n)", 1.1, 5.0, 1.5)
        a = st.sidebar.slider("Scale (a)", 10, 200, 50)

    with col2:
        time_steps = np.arange(0, 300)
        block = gamma_block(time_steps, A, n, a)
        
        # Initialize head with h0
        total_head = np.full_like(time_steps, h0, dtype=float)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # Plot individual responses
        for p_t, p_a in pulses:
            # Shift and scale the block response
            resp = np.zeros_like(time_steps, dtype=float)
            if p_t < len(time_steps):
                valid_len = len(time_steps) - p_t
                resp[p_t:] = block[:valid_len] * p_a
                ax2.plot(time_steps, resp + h0, alpha=0.4, linestyle=':')
                total_head += resp
            
            ax1.bar(p_t, p_a, color='blue', width=2)

        ax1.set_ylabel("Rainfall (mm)")
        ax1.set_title("Input Pulses")
        
        ax2.plot(time_steps, total_head, color='black', lw=2, label="Total Simulated Head")
        ax2.axhline(h0, color='gray', linestyle='--', label="Baseline (h0)")
        ax2.set_ylabel("Groundwater Head (m)")
        ax2.set_title("Resulting Head (Sum of Responses)")
        ax2.legend()
        
        st.pyplot(fig)