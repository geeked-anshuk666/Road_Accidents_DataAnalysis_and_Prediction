# US Accidents Severity Prediction

This repository contains a machine learning project focused on analyzing and predicting the severity of road accidents in the United States using the `US_Accidents_March23.csv` dataset. The dataset includes detailed information about road accidents, including factors such as weather conditions, location, time, and accident severity. The objective of this project is to build a machine learning model to predict the severity of accidents based on these factors.

## Project Overview

The goal of this project is to predict the severity of accidents using various features such as weather conditions, road conditions, and time of day. The key steps involved are data cleaning, feature engineering, exploratory data analysis (EDA), and training machine learning models.

### Key Features of the Dataset:

- **ID**: Unique identifier for each accident record.
- **Severity**: The severity of the accident (1-4 scale).
- **Start_Time & End_Time**: Timestamps of when the accident started and ended.
- **Location**: Latitude and longitude of the accident.
- **Weather and Environmental Factors**: Temperature, humidity, wind speed, pressure, visibility, and precipitation.
- **Time-Related Features**: Hour of the day, weekday, month, and whether the accident occurred during peak hours.

### Expected Outcome

- **Jupyter Notebook/Google colab/Python file**: The analysis, data processing, feature engineering, and model training are conducted within a Jupyter notebook/ google colab notebook or Python file.
- **ML Model**: A Random Forest Classifier model is used to predict the severity of accidents.
- **Recommendations**: Data-driven recommendations to improve road safety, such as focusing on peak hours and improving response times during adverse weather conditions.

## Installation

To run this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/your-username/accident-severity-prediction.git
cd accident-severity-prediction
pip install -r requirements.txt
```
## Python Libraries Used
- `pandas` – Data manipulation and analysis
- `seaborn` – Data visualization
- `matplotlib` – Plotting and graphing
- `scikit-learn` – Machine learning algorithms, model evaluation, and data splitting
- `LabelEncoder` – Encoding categorical variables

## Data Processing & Feature Engineering

The following steps were taken to preprocess and engineer features from the raw dataset:

1. **Datetime Parsing**: The `Start_Time` and `End_Time` columns were converted to datetime objects for easier manipulation.
2. **Data Cleaning**: Removed irrelevant columns like `Country`, and filled missing values in weather-related columns using their mean values.
3. **Feature Engineering**:
   - **Accident Duration**: Calculated the duration of accidents in minutes.
   - **Peak Hours**: Flagged whether the accident occurred during peak traffic hours (6-9 AM and 4-7 PM).
   - **Adverse Weather**: Flagged whether the accident occurred during adverse weather conditions (rain, snow, fog, storm).
   - **Day & Month**: Extracted the weekday and month of the accident for further analysis.

## Exploratory Data Analysis (EDA)

Several visualizations were created to gain insights into the dataset:

1. **Accidents by Hour**: A line plot showing the number of accidents occurring in each hour of the day.
2. **Severity and Weather Conditions**: A stacked bar chart comparing accident severity during adverse weather conditions.
3. **Accidents by Weekday**: A bar plot showing accident frequency by weekday.
4. **Severe Accidents with Long Duration**: A bar chart analyzing the severity of accidents that lasted more than 120 minutes.

## Machine Learning Model

### Data Preprocessing

- **Features Used for Prediction**: The model is trained using various weather and time-related features like temperature, humidity, wind speed, visibility, and accident duration.
- **Label Encoding**: The target variable `Severity` is encoded using label encoding to convert categorical values into numerical labels.
- **Train-Test Split**: The dataset was split into training (75%), validation (15%), and testing (10%) sets to ensure proper evaluation of the model.

### Model Training

A **Random Forest Classifier** was used to predict the severity of accidents based on the processed features. Hyperparameters like `n_estimators=100` were used to configure the model.

### Model Evaluation

The model was evaluated using:
- **Classification Report**: Precision, recall, f1-score, and support for each class.
- **Accuracy Score**: Overall accuracy of the model on the test set.
- **Feature Importance**: A bar plot showing the importance of each feature in predicting accident severity.

## Results

The Random Forest model performed well, providing meaningful insights into the factors that influence accident severity. Some of the key findings include:
- Peak hours (6-9 AM, 4-7 PM) and adverse weather conditions are strongly correlated with severe accidents.
- Long-duration accidents tend to be more severe, highlighting the need for faster emergency response systems.

## Recommendations

Based on the analysis, the following strategies are recommended to mitigate accidents:
1. Focus resources on peak hours (6-9 AM, 4-7 PM) to reduce accidents.
2. Enhance road safety measures during adverse weather conditions (rain, snow, fog).
3. Deploy additional traffic patrols on weekends and high-risk weekdays.
4. Improve rapid response systems for long-duration severe accidents.

