# 🌱 Sustainability Machine Learning

A Python-based machine learning tool designed to predict company sustainability scores based on carbon emission standards and corporate commitments.

## 📊 Overview

This project demonstrates how to build a machine learning model that predicts sustainability scores using data about companies' carbon emission targets. It leverages pandas for data manipulation and PyTorch for neural network modeling.

## 🔑 Key Features

- 📝 **Data Processing**: Clean and encode categorical data
- 🧮 **Feature Engineering**: Extract meaningful features from text descriptions
- 🤖 **Machine Learning Pipeline**: Build and train a neural network model
- ✨ **Prediction System**: Generate sustainability scores based on carbon commitments

## ⚙️ Technologies Used

- 🐼 **pandas**: Data manipulation and analysis
- 🧠 **PyTorch**: Neural network implementation and training
- 🐍 **Python**: Core programming language

## 📋 Data Structure

The tool processes company data including:
- 🏢 Company names (encoded as category codes)
- 📅 Reporting year (normalized to 0)
- 🌍 Carbon focus type (Carbon Negative or Carbon Neutral)
- 🎯 Target year for carbon commitments
- 📈 Sustainability scores (normalized between 0-1)

## 🧪 Methodology

### 📊 Data Preprocessing
1. 📖 **Text Parsing**: Extract carbon focus type and target year from text descriptions
2. 🔢 **Categorical Encoding**: Convert company names and focus types to numerical codes
3. 📏 **Normalization**: Scale numerical features to improve model performance

### 🤖 Model Architecture
The neural network uses the processed data to learn patterns between carbon commitments and sustainability scores, enabling predictions for new companies.

## ▶️ Usage

Run `finance.py` to:
1. 📥 Load sample company data
2. 🔍 Process and encode features
3. 📦 Convert data to PyTorch tensors
4. 💡 Display results including encoded DataFrame and tensor representations

## 🎓 Learning Outcomes

This project demonstrates:
- 🧹 Data cleaning and preprocessing techniques
- 📚 Feature extraction from unstructured text
- 🎨 Categorical data encoding methods
- 📊 PyTorch tensor creation and manipulation
- 🤖 Machine learning model preparation pipeline
- 📉 Data normalization for improved model performance

## 🔮 Future Enhancements

- 📈 Add more comprehensive company datasets
- 🧪 Implement cross-validation for better model evaluation
- 🌐 Extend to include additional sustainability metrics
- 🌐 Deploy as a web application for broader accessibility