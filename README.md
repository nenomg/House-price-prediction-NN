



# House-price-prediction-NN


## House Price Prediction with Neural Networks

This repository contains code and data for a project that predicts house prices using neural networks. It includes data preprocessing, visualization of data relationships, model creation, training, and evaluation.

---

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Creation](#model-creation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

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
   git clone https://github.com/yourusername/house-price-prediction.git
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

---

## Exploratory Data Analysis

This section includes functions for visualizing the correlation matrix and scatter plots of house prices.

---

## Model Creation

A neural network model is created using TensorFlow's Keras API. It includes an input layer, hidden layers, and an output layer for regression.
![image](https://github.com/nenomg/House-price-prediction-NN/assets/105873794/713b4102-2493-4a09-a61c-bdd94e98dde0)

---

## Model Training

The model is trained using the training data, and the training loss is monitored. The trained model is used to make predictions.

---

## Evaluation

The evaluation section includes visualizations of the training and validation loss during model training, as well as scatter plots comparing actual and predicted house prices.

---

## Results

The project's results are presented in the form of visualizations and predictions made by the trained neural network model.

---

## Contributing

Contributions are welcome! If you have suggestions, bug reports, or want to contribute to this project, please open an issue or create a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

Feel free to explore the code and use it for your own house price prediction project. If you have any questions or need further assistance, please don't hesitate to reach out.

Happy coding! 🏡💰
