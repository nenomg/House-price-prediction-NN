



# House-price-prediction-NN


## House Price Prediction with Neural Networks

This repository contains code and data for a project that predicts house prices using neural networks. It includes data preprocessing, visualization of data relationships, model creation, training, and evaluation.

---

## Introduction

This project aims to predict house prices using a neural network model. It includes data preprocessing, visualization of data relationships, model creation, training, and evaluation. The code is written in Python and uses popular libraries such as Pandas, Seaborn, Matplotlib, and TensorFlow.

---

## Getting Started

### Prerequisites

Make sure you have the following prerequisites installed:

- Python 3.x
- Pandas
- Seaborn
- Matplotlib
- Scikit-Learn
- TensorFlow

### Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/nenomg/House-price-prediction-NN.git
   ```

2. Change directory to the project folder:

   ```bash
   cd house-price-prediction
   ```

3. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. **Load the Dataset**: Make sure you have the dataset `kc_house_data.csv` in the project directory.

2. **Run the Code**: Execute the code in `house_price_prediction.py` to perform data preprocessing, model creation, training, and evaluation.

---

## Data

The dataset used in this project is `kc_house_data.csv`, which contains information about house sales in King County, Washington, USA.

---

## Data Preprocessing

The data preprocessing steps include loading the dataset, removing unnecessary columns, and filtering rows based on correlation values.

![image](https://github.com/nenomg/House-price-prediction-NN/assets/105873794/c309baa8-cd03-4f25-b264-76f467cff4c9)

This is the correlation matrix for all of the data, we're going to remove every column that has a correlation of 0.1 or less with the price variable. The resultant matrix is the folowing:

![image](https://github.com/nenomg/House-price-prediction-NN/assets/105873794/aaae3516-7bc0-43ca-a62c-8be37fae8303)


---

## Exploratory Data Analysis

This section includes functions for visualizing scatter plots of house prices.

![image](https://github.com/nenomg/House-price-prediction-NN/assets/105873794/ea1a9086-dbca-4d5b-ba1a-39aef0352187)


---

## Model Creation

A neural network model is created using TensorFlow's Keras API. It includes an input layer, hidden layers, and an output layer for regression.

![image](https://github.com/nenomg/House-price-prediction-NN/assets/105873794/713b4102-2493-4a09-a61c-bdd94e98dde0)

---

## Model Training

The model is trained using the training data, and the training loss is monitored. The trained model is used to make predictions. We extract the data for test and train, then we scale the values.

![image](https://github.com/nenomg/House-price-prediction-NN/assets/105873794/5e7b8f3d-0934-41c9-8d57-7cc53c92cb64)


---

## Evaluation

The evaluation section includes visualizations of the training and validation loss during model training, as well as scatter plots comparing actual and predicted house prices.

### 10 EPOCHS

![image](https://github.com/nenomg/House-price-prediction-NN/assets/105873794/d0a10c77-953a-450c-81c0-372235bf9c99)


### 50 EPOCHS

![image](https://github.com/nenomg/House-price-prediction-NN/assets/105873794/1006307f-c351-4871-8634-95118cd083f1)


### 200 EPOCHS

![image](https://github.com/nenomg/House-price-prediction-NN/assets/105873794/ac6efe8b-739c-4152-89c4-d9ee35645266)


---

## Results

The project's results are presented in the form of visualizations and predictions made by the trained neural network model.

### 10 EPOCHS

![image](https://github.com/nenomg/House-price-prediction-NN/assets/105873794/3e53086b-f63b-438e-b08a-5a1b9753916e)


### 50 EPOCHS

![image](https://github.com/nenomg/House-price-prediction-NN/assets/105873794/82a73bf2-1358-4a92-ac58-3d509bd8982f)


### 200 EPOCHS

![image](https://github.com/nenomg/House-price-prediction-NN/assets/105873794/2ff12a19-743f-4020-b7c8-45cefbbf9d5e)

---

Feel free to explore the code and use it for your own house price prediction project. If you have any questions or need further assistance, please don't hesitate to reach out.

Happy coding! 🏡💰
