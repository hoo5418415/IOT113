# **ChatGpt Prompt **

1.請根據上傳的csv檔，使用auto regression，根據Date與Price進行model evaluation 
做完評估後，輸出預測漏斗圖，以x軸為Date,y軸為Price,以預測未來三個月個價格


2.can you generate the python code about how to predict the value?

3.please generate the markdown format so I can upload to github as read.me document

---
# **CRISP-DM Framework Analysis**

## 1. Business Understanding

### Objective:
Predict the future prices of a certain asset (e.g., stock, commodity) based on historical "Date" and "Price" data using an Auto Regressive (AR) model. The prediction will help in making informed decisions for the upcoming months (specifically a 3-month forecast).

### Key Business Questions:
- What are the expected price trends over the next three months?
- How reliable is the model in predicting future prices based on historical data?

---

## 2. Data Understanding

### Data Overview:
- **File:** `2317_112,113.csv`
- **Columns:**
  - `Date`: Dates in ROC (Taiwanese) calendar format.
  - `Price`: Asset prices corresponding to the dates.
  
### Initial Observations:
- The `Date` column follows the ROC calendar, which needs to be converted to the Gregorian calendar for ease of analysis.
- The `Price` data appears to be continuous and suitable for time series analysis.

### Key Insights:
- The dataset spans from early 2023, indicating recent data.
- No missing values were observed in the sample.

---

## 3. Data Preparation

### Steps:
1. **Convert Date Format:** 
   The ROC calendar was converted to the Gregorian calendar for better compatibility with Python's time series libraries.
   
   ```python
   def convert_roc_to_gregorian(roc_date):
       roc_year, month, day = map(int, roc_date.split('/'))
       gregorian_year = roc_year + 1911
       return datetime(gregorian_year, month, day)
	```
2. **Indexing:**
  The Date column was set as the index to enable time series analysis:
  ```python
     data.set_index('Date', inplace=True)
  ```

3. **Split Data:**
  The data was split into training and testing sets:

  Training Set: All data except the last 30 days.
  Test Set: The last 30 days of data.

4. **Auto Regressive (AR) Model:**
  An AR model with a lag of 5 days was fitted to the training data.

---

## 4.Modeling
**Chosen Model:**
Auto Regressive (AR) Model: The AR model is suitable for univariate time series forecasting, assuming that future values are linearly dependent on previous values.

**Model Training:**
The AR model was trained on historical Price data using 5 lags, meaning that each prediction was based on the past 5 days of data.

**Model Evaluation:**
The model was evaluated using the Root Mean Squared Error (RMSE) metric, yielding an RMSE of approximately 9.71, which represents the average error between predicted and actual prices in the test set.

---

## 5.Evaluation
**Model Performance:**
The RMSE of 9.71 indicates that on average, the model’s predictions deviate by around 9.71 units from the actual values.

**Visual Validation:**
A visual comparison of actual prices vs. predicted prices for the next three months was plotted. The model follows the general trend of price movement but may need further tuning to reduce error.

---

## 6. Deployment

**Prediction for the Next 3 Months:**
The AR model was used to forecast prices for the next 90 business days (3 months). The forecast was visualized in a line plot.

**Next Steps:**
Continue monitoring the model’s predictions against actual values as new data becomes available.
Explore more advanced models like ARIMA or SARIMA to potentially improve forecast accuracy.

---

## 7. Conclusion
The current AR model provides a reasonable forecast with an acceptable level of error (RMSE = 9.71), but further refinement could enhance accuracy.

This model can serve as a foundation for more sophisticated predictive modeling, including incorporating external variables or moving to more complex time series models.


![image](https://github.com/user-attachments/assets/0ecff37f-2e7a-4eee-ae73-23ba5837ceec)

