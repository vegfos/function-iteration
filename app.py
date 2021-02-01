import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from function_iteration import *

def sidebar():
    st.sidebar.header('Parameters:')
    b = st.sidebar.number_input('Beta (Discount Rate)', 
        min_value = 0.0,
        max_value = 1.0,
        value = 0.97,
        step =  0.01)

    r = st.sidebar.number_input('R (Interest Rate)', 
        min_value = 1.000,
        max_value = 5.000,
        value = 1.015,
        step =  0.001)

    u_func = st.sidebar.radio('Utility function', 
        options = ['log'])

    if u_func == 'log':
        u = lambda c : np.log(c)

    st.sidebar.header('Numerics (State Space):')

    st.sidebar.header('State Space')
    w_low, w_high = st.sidebar.slider('Discrete States',
        value=(0, 1000))
    
    n = st.sidebar.slider('State steps', 
        min_value = 5,
        max_value = 5000,
        value = 100,
        step =  1)

    st.sidebar.header('Control Space')
    c_low, c_high = st.sidebar.slider('Discrete Controls',
        max_value= 100,
        value=(0, 50))

    m = st.sidebar.slider('Control steps', 
        min_value = 5,
        max_value = 5000,
        value = 100,
        step =  1)

    st.sidebar.header('Stopping Criteria')
    maxit = st.sidebar.number_input('Max Iterations',
        min_value = 50,
        max_value = 1000,
        value = 400)

    tol = 10e-2

    return b, r, u, w_low, w_high, n, c_low, c_high, m, maxit, tol

def simulate_time_path(years, w0, wealth, policy):
    """ Runs a time a path simulation for a policy
    """
    policy_path = []
    wealth_path = []
    years = years 

    for i in range(years):
        policy_index = np.argmin(np.abs(np.subtract.outer(w0, wealth)))
        control = policy[policy_index]

        policy_path.append(control)
        w0 = w0 - control
        wealth_path.append(w0)

    simul_df = pd.DataFrame(data={
            'Wealth': wealth_path,
            'Policy': policy_path
        })
    
    return simul_df

def wealth_simulation_chart(df):
    chart = alt.Chart(df.reset_index()).mark_line().encode(
            x='index',
            y='Wealth'
            ).properties(
                title='Time Path Wealth'
            )
    return chart

def policy_simulation_chart(df):
    chart = alt.Chart(df.reset_index()).mark_line().encode(
            x='index',
            y='Policy'
        ).properties(
            title='Time Path Consumption'
        )

    return chart

@st.cache(suppress_st_warning=True)
def run_function_iteration(b, r, u, w_low, w_high, n, c_low, c_high, m, maxit, tol):
    value_function, wealth, policy = function_iteration(b, r, u, w_low, w_high, n, c_low, c_high, m, maxit, tol)

    return value_function, wealth, policy

run_button = st.sidebar.empty()
value_function = None
b, r, u, w_low, w_high, n, c_low, c_high, m, maxit, tol = sidebar()

st.title('Function Iteration')

value_function, wealth, policy = run_function_iteration(b, r, u, w_low, w_high, n, c_low, c_high, m, maxit, tol)

data = pd.DataFrame({'Value Function' : value_function,
        'Wealth' : wealth, 
        'Policy' : policy}).replace(-np.inf, np.nan).dropna()

value_function_chart = alt.Chart(data).mark_line().encode(
        x='Wealth',
        y='Value Function'
        ).properties(
            title='Value Function'
    )

policy_chart = alt.Chart(data).mark_line().encode(
        x='Wealth',
        y='Policy'
        ).properties(
            title='Control Rule'
        )

col1, col2 = st.beta_columns(2)
    
with col1:
    st.write(value_function_chart)
    
with col2:
    st.write(policy_chart)

st.subheader('Time Path Simulation')
col3, col4 = st.beta_columns(2)
with col3:
    w0 = st.slider('Initial Wealth', 
    min_value = 100,
    max_value = 10000,
        value = 1000)

with col4:
    years = st.slider('Number of years',
        min_value = 5,
        max_value = 500,
        value=100)


def simulation(years, w0, wealth, policy):
    simul_df = simulate_time_path(years, w0, wealth, policy)

    wealth_chart = wealth_simulation_chart(simul_df)
    policy_chart = policy_simulation_chart(simul_df)

    return wealth_chart, policy_chart

wealth_chart, policy_chart = simulation(years, w0, wealth, policy)
col5, col6 = st.beta_columns(2)
with col5:
    st.write(wealth_chart)

with col6:
    st.write(policy_chart)