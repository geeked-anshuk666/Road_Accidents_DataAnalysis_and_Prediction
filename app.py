import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Loading  dataset
data = pd.read_csv("Dataset/US_Accidents_March23.csv")

# Convert time columns to datetime
data['Start_Time'] = pd.to_datetime(data['Start_Time'], errors='coerce', format='%Y-%m-%d %H:%M:%S.%f')
data['End_Time'] = pd.to_datetime(data['End_Time'], errors='coerce', format='%Y-%m-%d %H:%M:%S.%f')

# Drop rows with invalid datetime values
data.dropna(subset=['Start_Time', 'End_Time'], inplace=True)

# Data Cleaning
data.drop(['Country'], axis=1, inplace=True)
data['Temperature(F)'].fillna(data['Temperature(F)'].mean(), inplace=True)
data['Humidity(%)'].fillna(data['Humidity(%)'].mean(), inplace=True)

# Feature Engineering
data['Duration'] = (data['End_Time'] - data['Start_Time']).dt.total_seconds() / 60
data['Weekday'] = data['Start_Time'].dt.day_name()
data['Month'] = data['Start_Time'].dt.month_name()
data['Hour'] = data['Start_Time'].dt.hour
data['Peak_Hours'] = data['Hour'].apply(lambda x: 1 if (6 <= x <= 9 or 16 <= x <= 19) else 0)
data['Adverse_Weather'] = data['Weather_Condition'].apply(
    lambda x: 1 if any(y in str(x) for y in ['Rain', 'Snow', 'Fog', 'Storm']) else 0
)

# Visualization
# 1. Accidents by Hour
hourly_accidents = data.groupby('Hour').size()
sns.lineplot(x=hourly_accidents.index, y=hourly_accidents.values, color='red')
plt.title('Accidents by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Number of Accidents')
plt.xticks(range(0, 24))
plt.grid()
plt.show()

# 2. Severity and Weather Conditions
severity_weather = data.groupby(['Adverse_Weather', 'Severity']).size().unstack()
severity_weather.plot(kind='bar', stacked=True, figsize=(10, 6), colormap='coolwarm')
plt.title('Severity of Accidents During Adverse Weather')
plt.xlabel('Adverse Weather')
plt.ylabel('Number of Accidents')
plt.legend(title='Severity', loc='upper right')
plt.show()

# 3. Weekday Patterns
weekday_accidents = data['Weekday'].value_counts()
sns.barplot(x=weekday_accidents.index, y=weekday_accidents.values, palette='magma')
plt.title('Accidents by Weekday')
plt.xlabel('Weekday')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()

# 4. Accident Duration Analysis
long_durations = data[data['Duration'] > 120]
long_durations_by_severity = long_durations.groupby('Severity').size()
long_durations_by_severity.plot(kind='bar', color='purple')
plt.title('Severe Accidents with Long Duration')
plt.xlabel('Severity')
plt.ylabel('Number of Long-Duration Accidents')
plt.show()

# ML Model for Predicting Accident Severity
features = ['Start_Lat', 'Start_Lng', 'Temperature(F)', 'Humidity(%)',
            'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)',
            'Precipitation(in)', 'Duration', 'Hour', 'Peak_Hours', 'Adverse_Weather']
target = 'Severity'

data_ml = data[features + [target]].dropna()
label_encoder = LabelEncoder()
data_ml['Severity'] = label_encoder.fit_transform(data_ml['Severity'])

# Train-Validate-Test Split
X = data_ml[features]
y = data_ml['Severity']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

# Training Random Forest Classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Validation Performance
y_val_pred = model.predict(X_val)
print("Validation Performance:")
print(classification_report(y_val, y_val_pred))

# Test Performance
y_test_pred = model.predict(X_test)
print("Test Performance:")
print(classification_report(y_test, y_test_pred))
print("Accuracy Score:", accuracy_score(y_test, y_test_pred))

# Feature Importance
importances = model.feature_importances_
sns.barplot(x=importances, y=features, palette='viridis')
plt.title('Feature Importance for Severity Prediction')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

# Recommendations and strategies to mitigate accidents
print("### Recommendations ###")
print("1. Focus resources on peak hours (6-9 AM, 4-7 PM) to reduce accidents.")
print("2. Enhance road safety measures during adverse weather conditions (rain, snow, fog).")
print("3. Deploy additional traffic patrols on weekends and high-risk weekdays.")
print("4. Improve rapid response systems for long-duration severe accidents.")