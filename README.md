# Insight (Stock Price Prediction Model)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iEsMueqrB44LqXR_HbyDjqcg-DfZo8Ob)

## Overview

This project implements a Deep Learning model using Long Short-Term Memory (LSTM) networks to predict future stock prices for ADIB (Abu Dhabi Islamic Bank) based on historical market data.

The model is designed to learn temporal patterns from stock price movements and generate future price predictions using time-series forecasting techniques.

---

## Dataset

The model uses a cleaned dataset (`ADIB_Cleaned.csv`) containing historical stock market information, including:

* Date
* Open Price
* High Price
* Low Price
* Closing Price (Price)

### Data Processing

* Convert `Date` column to datetime format.
* Check for missing values.
* Perform exploratory data analysis (EDA).
* Visualize stock price trends over time.
* Normalize data using `MinMaxScaler`.
* Create time-series sequences using a rolling window of **100 previous trading days**.

---

## Model Architecture

The forecasting model is built using TensorFlow/Keras and consists of:

| Layer              | Configuration                   |
| ------------------ | ------------------------------- |
| LSTM Layer 1       | 50 units, Return Sequences=True |
| Dropout            | 0.2                             |
| LSTM Layer 2       | 50 units, Return Sequences=True |
| Dropout            | 0.2                             |
| LSTM Layer 3       | 50 units, Return Sequences=True |
| Dropout            | 0.2                             |
| LSTM Layer 4       | 50 units                        |
| Dropout            | 0.2                             |
| Dense Output Layer | 1 neuron                        |

### Compilation

* Optimizer: Adam
* Loss Function: Mean Squared Error (MSE)

---

## Training Configuration

* Epochs: 100
* Batch Size: 32
* Validation Data: Test Set

The model is trained on historical stock prices and validated using unseen data to evaluate forecasting performance.

---

## Evaluation Metrics

The model performance is evaluated using:

* R² Score
* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)

Additionally, prediction results are visualized against actual stock prices to compare forecasting accuracy.

---

## Workflow

1. Load and inspect stock market dataset.
2. Perform data cleaning and exploratory analysis.
3. Normalize price values.
4. Generate time-series sequences with a 100-day lookback window.
5. Train a stacked LSTM neural network.
6. Predict future stock prices.
7. Evaluate model performance.
8. Save the trained model as:

```bash
my_model.h5
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras

---

## Output

The trained model produces stock price predictions based on historical trends and can be used as a foundation for:

* Financial forecasting
* Stock market analysis
* Time-series prediction projects
* Deep learning research on financial datasets

---

## Future Improvements

* Incorporate additional technical indicators (RSI, MACD, Bollinger Bands).
* Use multi-feature forecasting instead of relying solely on closing prices.
* Apply hyperparameter optimization.
* Experiment with GRU, BiLSTM, or Transformer architectures.
* Implement real-time prediction pipelines.

## Model File

After training, the model is exported as:

```bash
my_model.h5
```

This file can be loaded later for inference without retraining the network.
