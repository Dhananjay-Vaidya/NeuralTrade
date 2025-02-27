AI-Powered Stock Trading System

# AI-Powered Stock Trading System  

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)
![Stable-Baselines3](https://img.shields.io/badge/RL-Stable%20Baselines3-orange)
![Yahoo Finance](https://img.shields.io/badge/Data-YahooFinance-blue)
![Last Commit](https://img.shields.io/github/last-commit/yourusername/ai-trading-bot)
![Issues](https://img.shields.io/github/issues/yourusername/ai-trading-bot)
![Forks](https://img.shields.io/github/forks/yourusername/ai-trading-bot?style=social)
![Stars](https://img.shields.io/github/stars/yourusername/ai-trading-bot?style=social)
![Contributions](https://img.shields.io/badge/contributions-welcome-blue)
![Build](https://img.shields.io/badge/build-passing-brightgreen)


Table of Contents

Project Description
Motivation
Problem Statement
Technologies Used
Current Features
Installation
Usage
Dataset
Contributing
Author
License


Project Description

The AI-Powered Stock Trading System is an advanced trading bot that uses Reinforcement Learning (PPO algorithm) to make automated buy/sell/hold decisions. The system is trained using historical stock data and deployed with a Streamlit dashboard for real-time monitoring of trading performance.


Motivation

Financial markets are complex, and making profitable trades requires intelligent decision-making. This project applies Deep Reinforcement Learning to simulate and execute optimal trading strategies without human intervention.


Problem Statement

The system is designed to address:
Automating trading strategies using AI & RL
Training an RL model to optimize trading performance
Evaluating the bot’s performance using historical stock data
Providing a Streamlit dashboard for trade visualization


Technologies Used
This project is built using:

Python – Core programming language
Stable-Baselines3 (PPO Algorithm) – Reinforcement Learning framework
Gym – Environment for RL model training
Yahoo Finance API (yfinance) – Fetching real-time stock data
TA (Technical Analysis) – Generating indicators (RSI, MACD, Bollinger Bands)
Pandas & NumPy – Data handling and preprocessing
Matplotlib & Plotly – Visualization of stock trends & trading actions
Streamlit – Web-based interactive dashboard


Current Features
Real-time stock data collection from Yahoo Finance
Feature engineering with advanced technical indicators
Trading environment simulation using Gym
Deep Reinforcement Learning (PPO Algorithm) for trading decisions
Trading performance evaluation using cumulative rewards
Streamlit dashboard for trade monitoring


Installation
To install and set up the project, follow these steps:

1️⃣ Clone the Repository

git clone https://github.com/Dhananjay-Vaidya/NeuralTrade.git
cd ai-trading-bot

2️⃣ Create a Virtual Environment 
python -m venv venv
source venv/bin/activate  
# On Windows: venv\Scripts\activat

3️⃣ Install Dependencies
pip install -r requirements.txt


Usage

1️⃣ Fetch Stock Data
python src/datacollection.py

2️⃣ Clean & Preprocess Data
python src/data_cleaning.py

3️⃣ Apply Feature Engineering
python src/feature_engineering.py

4️⃣ Train the RL Model
python src/train_rl_model.py

5️⃣ Evaluate the Model
python src/evaluation.py

6️⃣ Run the Streamlit Dashboard
streamlit run app.py

This will launch a web-based dashboard to visualize trading actions and portfolio growth.


Dataset

The system collects historical stock data using the Yahoo Finance API and preprocesses it for Reinforcement Learning.

Raw data: data/stock_data1.csv
Cleaned data: clean_stock_data.csv
Feature-engineered data: engineered_stock_data.csv


Contributing
Contributions are welcome! To contribute:

Fork the repository
Create a new branch (git checkout -b feature-name)
Make changes and commit (git commit -m "Description of changes")
Push to GitHub (git push origin feature-name)
Submit a pull request for review


License
This project is licensed under the MIT License. See the LICENSE file for details.

