# 💊 Rossmann Store Sales Forecasting with LSTM

## 🌟 1. Project Overview

This project implements a deep learning solution for multivariate time-series forecasting of daily sales data from the Rossmann drug stores (2013-2015). The goal is to accurately predict future sales to optimize operational decisions.

 Model: Stacked LSTM (Long Short-Term Memory) Neural Network.
 Dataset: Rossmann Store Sales (Focus: Data for Store 1).

---

## ⚙️ 2. Methodology and Technical Stack

### 2.1 Data Processing

| Step | Detail |
| :--- | :--- |
| Data Cleaning | Handled missing values (NaN replaced with 0). Filtered out all days when the store was closed (`Open` = 0). |
| Feature Engineering | Extracted temporal features: `Month`, `Year`, `DayOfWeek`, and `WeekOfYear`. |
| Feature Scaling | Features were scaled using MinMaxScaler for optimal LSTM performance. |
| Sequence Creation | Data was reshaped into sequences of 30 timesteps (days) to serve as the look-back window for the model. |

### 2.2 Model Architecture

The model uses a Stacked LSTM design, effective for capturing long-term dependencies in sequential data.

| Layer Type | Units/Size | Notes |
| :--- | :--- | :--- |
| LSTM (Layer 1) | 100 | Captures sequential patterns. |
| LSTM (Layer 2) | 100 | Captures deeper temporal features. |
| Dropout | 20% | Applied after each LSTM layer for regularization. |
| Output | 1 | Final prediction of the next day's Sales. |
| Optimization | Adam optimizer, Loss: MSE (Mean Squared Error). |

---

## 🔬 3. Model Performance and Evaluation

The model showed excellent performance on the unseen test data:

### 3.1 Key Metrics

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| R² Score | $\mathbf{> 0.85}$ | The model explains over $85\%$ of the variance in sales. |
| RMSE | $\approx \mathbf{1,215}$ | Root Mean Squared Error (deviation in Euros). |
| MAE | $\approx \mathbf{705}$ | Mean Absolute Error (average error in Euros). |

### 3.2 Prediction Quality

 Training Dynamics: Validation Loss tracked the Training Loss closely, confirming the model generalized well.
 Prediction vs. Actual: The scatter plot showed predictions clustering tightly around the ideal $45^\circ$ line, indicating low residual error across the sales range.

---

## 🔮 4. Future Forecasting

The trained model generated a stable $\mathbf{30}$-day forecast (the dashed line on the final plot) that smoothly continues the historical sales trend, providing a reliable baseline for the store's sales outlook.



---

## 🛠️ 2. Installation and Setup (For Jupyter Notebook)

To run the accompanying `.ipynb` notebook, install the necessary libraries using the following command in a notebook cell or your terminal:

### 2.1 Create env file
``` bash
python -m venv sales_env


### 2.2 Install Libraries

```bash
pip install -r requirements.txt


