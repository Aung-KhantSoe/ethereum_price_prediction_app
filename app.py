import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
import streamlit as st

import streamlit as st

# Initialize session state for configuration visibility and thresholds
if "show_config" not in st.session_state:
    st.session_state.show_config = False

if "increase_range" not in st.session_state:
    st.session_state.increase_range = (0, 5)  # Default Increase range (0% to 5%)
if "large_increase_range" not in st.session_state:
    st.session_state.large_increase_range = (5, 100)  # Default Large Increase range (> 5% to 20%)
if "decrease_range" not in st.session_state:
    st.session_state.decrease_range = (-5, 0)  # Default Decrease range (-5% to 0%)
if "large_decrease_range" not in st.session_state:
    st.session_state.large_decrease_range = (-100, -5)  # Default Large Decrease range (< -5% to -20%)

# Default values for thresholds
DEFAULT_INCREASE_RANGE = (0, 5)  # Default Increase range (0% to 5%)
DEFAULT_LARGE_INCREASE_RANGE = (5, 100)  # Default Large Increase range (> 5% to 20%)
DEFAULT_DECREASE_RANGE = (-5, 0)  # Default Decrease range (-5% to 0%)
DEFAULT_LARGE_DECREASE_RANGE = (-100, -5)  # Default Large Decrease range (< -5% to -20%)

# Button to toggle configuration visibility
if st.sidebar.button("Configure Thresholds"):
    st.session_state.show_config = not st.session_state.show_config  # Toggle visibility

# Show configuration sliders if visibility is toggled on
if st.session_state.show_config:
    with st.sidebar.expander("Adjust Percentage Thresholds", expanded=True):
        # Increase and Large Increase (connected range sliders)
        st.session_state.increase_range = st.slider(
            "Increase Range (%)", 
            min_value=0, 
            max_value=20, 
            value=st.session_state.increase_range)  # Use session state value
        
        # Ensure Large Increase Range starts at the end of Increase Range
        st.session_state.large_increase_range = st.slider(
            "Large Increase Range (%)", 
            min_value=st.session_state.increase_range[1],  # Large Increase starts at the end of Increase Range
            max_value=100, 
            value=(st.session_state.increase_range[1], st.session_state.large_increase_range[1]))  # Enforce connection

        # Decrease and Large Decrease (connected range sliders)
        st.session_state.decrease_range = st.slider(
            "Decrease Range (%)", 
            min_value=-20, 
            max_value=0, 
            value=st.session_state.decrease_range)  # Use session state value
        
        # Ensure Large Decrease Range ends at the start of Decrease Range
        st.session_state.large_decrease_range = st.slider(
            "Large Decrease Range (%)", 
            min_value=-100, 
            max_value=st.session_state.decrease_range[0],  # Large Decrease ends at the start of Decrease Range
            value=(st.session_state.large_decrease_range[0], st.session_state.decrease_range[0]))  # Enforce connection

        # Reset Button
        if st.button("Reset to Default"):
            st.session_state.increase_range = DEFAULT_INCREASE_RANGE
            st.session_state.large_increase_range = DEFAULT_LARGE_INCREASE_RANGE
            st.session_state.decrease_range = DEFAULT_DECREASE_RANGE
            st.session_state.large_decrease_range = DEFAULT_LARGE_DECREASE_RANGE
            st.success("Thresholds reset to default values!")

# Function to get detailed state based on user-defined thresholds
def get_detailed_state(change):
    if st.session_state.large_increase_range[0] <= change <= st.session_state.large_increase_range[1]:
        return 'Large Increase'
    elif st.session_state.increase_range[0] <= change <= st.session_state.increase_range[1]:
        return 'Increase'
    elif st.session_state.decrease_range[0] <= change <= st.session_state.decrease_range[1]:
        return 'Decrease'
    elif st.session_state.large_decrease_range[0] <= change <= st.session_state.large_decrease_range[1]:
        return 'Large Decrease'
    else:
        return 'No Change'  # For values outside the defined ranges


# Function to create a first-order transition matrix
def create_first_order_transition_matrix(states, state_order):
    transitions = {}
    for i in range(len(states) - 1):
        key = states[i]
        if key not in transitions:
            transitions[key] = []
        transitions[key].append(states[i + 1])
    
    transition_matrix = {}
    for key, next_states in transitions.items():
        state_counts = pd.Series(next_states).value_counts(normalize=True)
        transition_matrix[key] = state_counts
    
    # Ensure all states are present
    for state in state_order:
        if state not in transition_matrix:
            transition_matrix[state] = pd.Series(0, index=state_order)
        else:
            for next_state in state_order:
                if next_state not in transition_matrix[state]:
                    transition_matrix[state][next_state] = 0
    
    return transition_matrix

# Function to create a second-order transition matrix
def create_second_order_transition_matrix(states, state_order):
    transitions = {}
    for i in range(len(states) - 2):
        key = (states[i], states[i + 1])
        if key not in transitions:
            transitions[key] = []
        transitions[key].append(states[i + 2])
    
    transition_matrix = {}
    for key, next_states in transitions.items():
        state_counts = pd.Series(next_states).value_counts(normalize=True)
        transition_matrix[key] = state_counts
    
    # Ensure all combinations are present
    for state1 in state_order:
        for state2 in state_order:
            key = (state1, state2)
            if key not in transition_matrix:
                transition_matrix[key] = pd.Series(0, index=state_order)
            else:
                for state in state_order:
                    if state not in transition_matrix[key]:
                        transition_matrix[key][state] = 0
    
    return transition_matrix

# Function to predict the next state for first-order Markov
def predict_next_state_first_order(current_state, transition_df):
    if current_state in transition_df.index:
        next_state_probabilities = transition_df.loc[current_state]
        next_state = next_state_probabilities.idxmax()
        return next_state
    else:
        return 'Unknown'

# Function to predict the next state for second-order Markov
def predict_next_state_second_order(previous_state, current_state, transition_df):
    if (previous_state, current_state) in transition_df.index:
        next_state_probabilities = transition_df.loc[(previous_state, current_state)]
        next_state = next_state_probabilities.idxmax()
        return next_state
    else:
        return 'Unknown'

# Function to predict future states using a first-order Markov Chain
def predict_future_states_first_order(transition_matrix, current_state, days):
    states = [current_state]
    for _ in range(days):
        key = states[-1]
        if key in transition_matrix:
            next_state = np.random.choice(transition_matrix[key].index, p=transition_matrix[key].values)
        else:
            next_state = np.random.choice(state_order)
        states.append(next_state)
    return states

# Function to predict future states using a second-order Markov Chain
def predict_future_states_second_order(transition_matrix, current_state, days):
    states = list(current_state)
    for _ in range(days):
        key = tuple(states[-2:])
        if key in transition_matrix:
            next_state = np.random.choice(transition_matrix[key].index, p=transition_matrix[key].values)
        else:
            next_state = np.random.choice([key[-1] for key in transition_matrix.keys()])
        states.append(next_state)
    return states

# Streamlit app
st.title("Ethereum Price State Prediction")

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data['Date'] = pd.to_datetime(data['Date'])
    # Filter data to start from 2020
    # data = data[data['Date'] >= '2020-01-01']

    # Show sample dataset
    st.subheader("Sample Dataset")

    # Calculate daily price changes
    data['Price_Change'] = data['Close'] - data['Open']
    data['Price_Change_Percentage'] = ((data['Close'] - data['Open']) / data['Open']) * 100

    # Drop the first row with NaN value
    data = data.dropna().reset_index(drop=True)

    # Apply the detailed state function
    data['Detailed_State'] = data['Price_Change_Percentage'].apply(get_detailed_state)
    st.write(data.head())

    # Define the custom order for the states
    state_order = ['Large Decrease', 'Decrease', 'Increase', 'Large Increase']

    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    model_type = st.sidebar.radio("Choose Markov Model", ["1-State Markov", "2-State Markov"])

    if model_type == "1-State Markov":
        # Create the first-order transition matrix
        states = data['Detailed_State'].dropna().values
        transition_matrix = create_first_order_transition_matrix(states, state_order)

        # Convert the transition matrix to a DataFrame for visualization
        transition_df = pd.DataFrame(transition_matrix).fillna(0).T

        # Ensure all states are included in the DataFrame
        for state in state_order:
            if state not in transition_df.index:
                transition_df.loc[state] = [0] * len(state_order)
        transition_df = transition_df[state_order]

        # Plot the heatmap
        st.subheader("First-Order Transition Matrix Heatmap")
        plt.figure(figsize=(12, 8))
        sns.heatmap(transition_df, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('First-Order Transition Matrix Heatmap')
        plt.xlabel('Next State')
        plt.ylabel('Current State')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(plt)

        # Split the dataset into training and test sets
        train_data, test_data = train_test_split(data, test_size=0.3, shuffle=False)

        # Create the first-order transition matrix using the training set
        train_states = train_data['Detailed_State'].dropna().values
        transition_matrix = create_first_order_transition_matrix(train_states, state_order)

        # Convert the transition matrix to a DataFrame for easier access
        transition_df = pd.DataFrame(transition_matrix).fillna(0).T

        # Ensure all states are included in the DataFrame
        for state in state_order:
            if state not in transition_df.index:
                transition_df.loc[state] = [0] * len(state_order)
        transition_df = transition_df[state_order]

        # Test the accuracy of the predictions
        correct_predictions = 0
        total_predictions = 0

        for i in range(1, len(test_data) - 1):
            current_state = test_data['Detailed_State'].iloc[i - 1]
            actual_next_state = test_data['Detailed_State'].iloc[i]
            
            predicted_next_state = predict_next_state_first_order(current_state, transition_df)
            
            if predicted_next_state == actual_next_state:
                correct_predictions += 1
            total_predictions += 1

        accuracy = correct_predictions / total_predictions

        # Display model accuracy
        st.subheader("Model Accuracy")
        st.markdown(f"""
        <div style="background-color:#f0f2f6;padding:10px;border-radius:10px;">
            <h3 style="color:#1f77b4;text-align:center;">Accuracy: {accuracy:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Predict next 30 days' states
        st.subheader("Next 30 Days Prediction")
        current_state = data['Detailed_State'].iloc[-1]
        predicted_states = predict_future_states_first_order(transition_matrix, current_state, 30)

        # Create a DataFrame for visualization
        future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=30, freq='D')
        predicted_df = pd.DataFrame({'Date': future_dates, 'State': predicted_states[1:]})

        # Convert the 'State' column to a categorical type with the custom order
        predicted_df['State'] = pd.Categorical(predicted_df['State'], categories=state_order, ordered=True)

        # Visualize the predicted states
        st.write("Predicted States for the Next 30 Days:")
        st.write(predicted_df)

        # Plot the predicted states using Plotly
        fig = px.line(predicted_df, x='Date', y='State', title='Predicted Future States of Ethereum (ETH) Price', 
                      category_orders={'State': state_order})
        fig.update_yaxes(categoryorder='array', categoryarray=state_order)
        st.plotly_chart(fig)
        
        # Visualize historical data with states
        fig2 = px.line(data, x='Date', y='Close', color='Detailed_State', title='Historical Ethereum (ETH) Price with States', category_orders={'Detailed_State': state_order})
        fig2.update_yaxes(categoryorder='array', categoryarray=state_order)
        st.plotly_chart(fig2)

        # Sidebar for user input
        st.sidebar.header("Predict Next State")
        today_open_price = st.sidebar.number_input("Enter today's open price", value=0.0)
        today_close_price = st.sidebar.number_input("Enter today's close price", value=0.0)

        if st.sidebar.button("Predict"):
            # Determine the current state based on today's open and close prices
            change = (today_close_price - today_open_price) / today_open_price * 100
            current_state = get_detailed_state(change)

            # Predict the next state
            predicted_next_state = predict_next_state_first_order(current_state, transition_df)
            st.sidebar.subheader("Prediction Result")
            st.sidebar.write(f"Current state: {current_state}")
            st.sidebar.write(f"Predicted next state: {predicted_next_state}")

    elif model_type == "2-State Markov":
        # Create the second-order transition matrix
        states = data['Detailed_State'].dropna().values
        transition_matrix = create_second_order_transition_matrix(states, state_order)

        # Convert the transition matrix to a DataFrame for visualization
        transition_df = pd.DataFrame(transition_matrix).fillna(0).T

        # Ensure all state pairs and next states are included in the DataFrame
        for state1 in state_order:
            for state2 in state_order:
                if (state1, state2) not in transition_df.index:
                    transition_df.loc[(state1, state2)] = [0] * len(state_order)
        transition_df = transition_df[state_order]

        # Plot the heatmap
        st.subheader("Second-Order Transition Matrix Heatmap")
        plt.figure(figsize=(12, 8))
        sns.heatmap(transition_df, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Second-Order Transition Matrix Heatmap')
        plt.xlabel('Next State')
        plt.ylabel('Current State Pair')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        st.pyplot(plt)

        # Split the dataset into training and test sets
        train_data, test_data = train_test_split(data, test_size=0.3, shuffle=False)

        # Create the second-order transition matrix using the training set
        train_states = train_data['Detailed_State'].dropna().values
        transition_matrix = create_second_order_transition_matrix(train_states, state_order)

        # Convert the transition matrix to a DataFrame for easier access
        transition_df = pd.DataFrame(transition_matrix).fillna(0).T

        # Ensure all state pairs and next states are included in the DataFrame
        for state1 in state_order:
            for state2 in state_order:
                if (state1, state2) not in transition_df.index:
                    transition_df.loc[(state1, state2)] = [0] * len(state_order)
        transition_df = transition_df[state_order]

        # Test the accuracy of the predictions
        correct_predictions = 0
        total_predictions = 0

        for i in range(2, len(test_data) - 1):
            previous_state = test_data['Detailed_State'].iloc[i - 2]
            current_state = test_data['Detailed_State'].iloc[i - 1]
            actual_next_state = test_data['Detailed_State'].iloc[i]
            
            predicted_next_state = predict_next_state_second_order(previous_state, current_state, transition_df)
            
            if predicted_next_state == actual_next_state:
                correct_predictions += 1
            total_predictions += 1

        accuracy = correct_predictions / total_predictions

        # Display model accuracy
        st.subheader("Model Accuracy")
        st.markdown(f"""
        <div style="background-color:#f0f2f6;padding:10px;border-radius:10px;">
            <h3 style="color:#1f77b4;text-align:center;">Accuracy: {accuracy:.2%}</h3>
        </div>
        """, unsafe_allow_html=True)

        # Predict next 30 days' states
        st.subheader("Next 30 Days Prediction")
        current_state = (data['Detailed_State'].iloc[-2], data['Detailed_State'].iloc[-1])
        predicted_states = predict_future_states_second_order(transition_matrix, current_state, 30)

        # Create a DataFrame for visualization
        future_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=30, freq='D')
        predicted_df = pd.DataFrame({'Date': future_dates, 'State': predicted_states[2:]})

        # Convert the 'State' column to a categorical type with the custom order
        predicted_df['State'] = pd.Categorical(predicted_df['State'], categories=state_order, ordered=True)

        # Visualize the predicted states
        st.write("Predicted States for the Next 30 Days:")
        st.write(predicted_df)

        # Plot the predicted states using Plotly
        fig = px.line(predicted_df, x='Date', y='State', title='Predicted Future States of Ethereum (ETH) Price', 
                      category_orders={'State': state_order})
        fig.update_yaxes(categoryorder='array', categoryarray=state_order)
        st.plotly_chart(fig)
        
        # Visualize historical data with states
        fig2 = px.line(data, x='Date', y='Close', color='Detailed_State', title='Historical Ethereum (ETH) Price with States', category_orders={'Detailed_State': state_order})
        fig2.update_yaxes(categoryorder='array', categoryarray=state_order)
        st.plotly_chart(fig2)

        # Sidebar for user input
        st.sidebar.header("Predict Next State")
        today_open_price = st.sidebar.number_input("Enter today's open price", value=0.0)
        today_close_price = st.sidebar.number_input("Enter today's close price", value=0.0)
        previous_state = st.sidebar.selectbox("Enter the previous state", state_order)

        if st.sidebar.button("Predict"):
            # Determine the current state based on today's open and close prices
            change = (today_close_price - today_open_price) / today_open_price * 100
            current_state = get_detailed_state(change)

            # Predict the next state
            predicted_next_state = predict_next_state_second_order(previous_state, current_state, transition_df)
            st.sidebar.subheader("Prediction Result")
            st.sidebar.write(f"Current state: {current_state}")
            st.sidebar.write(f"Predicted next state: {predicted_next_state}")