# Ethereum Price State Prediction App

This is a Streamlit-based web application that predicts the future states of Ethereum (ETH) prices using 1st-order and 2nd-order Markov Chains. The app allows users to upload their dataset, configure state thresholds, and visualize predictions for the next 30 days.

## Table of Contents
- Cloning Process
- Streamlit Setup
- How the App Works
- How to Use the App
- About Markov Chains
- Features
- Future Improvements
- Contributing
- License

---

## Cloning Process
To get started with the app, clone the repository from GitHub and set up your environment:

### 1. Clone the Repository
```bash
git clone https://github.com/Aung-KhantSoe/ethereum_price_prediction_app.git
cd ethereum_price_prediction_app
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

---

## Streamlit Setup
To run this app locally, follow these steps:

### 1. Install Python
Ensure you have Python 3.7 or higher installed.

### 2. Install Required Libraries
```bash
pip install streamlit pandas numpy seaborn matplotlib plotly scikit-learn
```

### 3. Run the App
```bash
streamlit run app.py
```

### 4. Access the App
Open your browser and navigate to `http://localhost:8501`.

---

## How the App Works
The app uses Markov Chains to model transitions between different states of Ethereum price changes. The states are defined based on percentage changes in price:

- **Large Decrease**: Price decrease > 5%
- **Decrease**: Price decrease between 0% and 5%
- **Increase**: Price increase between 0% and 5%
- **Large Increase**: Price increase > 5%

### Key Components
- **Data Upload**: Users can upload a CSV file containing Ethereum price data (e.g., Date, Open, Close, High, Low).
- **State Configuration**: Users can adjust the thresholds for each state using sliders in the sidebar.
- **Transition Matrix**: The app calculates and visualizes the transition matrix for the selected Markov model (1st-order or 2nd-order).
- **Prediction**: The app predicts the next 30 days' states based on the selected model and displays the results in a table and a Plotly line chart.
- **Accuracy Calculation**: The app calculates the accuracy of the model by comparing predicted states with actual states in the test set.
- **Historical Trend Analysis**: Users can visualize past Ethereum price movements alongside their respective state transitions.

---

## How to Use the App

### 1. Upload Dataset
- Click on the **Upload your dataset (CSV file)** button and upload a CSV file containing Ethereum price data.
- The dataset should include columns like Date, Open, Close, High, and Low.

### 2. Configure Thresholds
- Click the **Configure Thresholds** button in the sidebar to adjust the percentage thresholds for each state.
- Use the sliders to define the ranges for Increase, Large Increase, Decrease, and Large Decrease.

### 3. Select Model
- Choose between **1-State Markov** and **2-State Markov** models using the radio button in the sidebar.

### 4. View Results
The app will display:
- A heatmap of the transition matrix.
- The accuracy of the model.
- Predicted states for the next 30 days in a table and a Plotly line chart.
- A historical price chart with states.

### 5. Predict Next State
- In the sidebar, enter today's open price, close price, and (if using 2-State Markov) the previous state.
- Click the **Predict** button to see the predicted next state.

---

## About Markov Chains

### What is a Markov Chain?
A Markov Chain is a statistical model that describes a sequence of events where the probability of each event depends only on the state of the previous event(s). It is widely used for modeling systems that transition between different states over time.

### 1st-Order Markov Chain
- The next state depends only on the current state.
- The transition matrix shows the probabilities of moving from one state to another.

### 2nd-Order Markov Chain
- The next state depends on the current state and the previous state.
- The transition matrix shows the probabilities of moving from a pair of states to the next state.

### Why Use Markov Chains for Price Prediction?
- Markov Chains are simple yet powerful for modeling sequential data like price changes.
- They provide insights into the likelihood of future states based on historical patterns.

---

## Features
- **Dynamic Threshold Configuration**: Users can adjust the percentage thresholds for each state.
- **Transition Matrix Visualization**: Heatmaps for 1st-order and 2nd-order transition matrices.
- **Next 30 Days Prediction**: Predicts future states and visualizes them in a table and a Plotly line chart.
- **Accuracy Calculation**: Evaluates the model's accuracy on a test set.
- **User-Friendly Interface**: Easy-to-use sidebar for configuration and prediction.
- **Historical Trend Analysis**: Displays past price movements and state transitions.
- **Downloadable Reports**: Export prediction results for further analysis.
- **State Probability Visualization**: Graphs displaying the probability of each state over time.

---

## Future Improvements
- **Add More Models**: Integrate additional models like ARIMA or LSTM for improved accuracy.
- **External Data Integration**: Include external factors like Bitcoin prices or news sentiment.
- **Real-Time Data**: Fetch real-time Ethereum price data using APIs.
- **Custom State Definitions**: Allow users to define custom states based on their preferences.
- **Advanced Visualization**: More interactive graphs and analytical tools.
- **Backtesting Feature**: Implement a backtesting system to compare predictions with actual price movements.

---

## Contributing
Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Enjoy using the app! If you have any questions or feedback, feel free to reach out. 🚀

