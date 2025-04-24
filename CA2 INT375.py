import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

sns.set(style="whitegrid")
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

file_path = "C:\\Users\\Priyanshu Singh\\Downloads\\Border_Crossing_Entry_Data.csv"
data = pd.read_csv(file_path)

data.columns = data.columns.str.strip()

print("First 5 rows:\n", data.head())

print("\nMissing values before cleaning:\n", data.isnull().sum())
 
for col in data.columns:
    if data[col].dtype == 'object':
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].mean(), inplace=True)

print("\nMissing values after cleaning:\n", data.isnull().sum())

data['Date'] = pd.to_datetime(data['Date'])

print("\nData Summary:\n", data.describe())

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Border Crossing EDA Visualizations', fontsize=20)

border_totals = data.groupby('Border')['Value'].sum().reset_index()
sns.barplot(x='Border', y='Value', data=border_totals, palette='Spectral', ax=axes[0, 0])
axes[0, 0].set_title('Total Crossings by Border')
axes[0, 0].set_ylabel('Total Crossings')
axes[0, 0].set_xlabel('Border Type')

sns.histplot(data['Value'], bins=30, kde=True, color='#F15BB5', ax=axes[0, 1])
axes[0, 1].set_title('Distribution of Crossing Values')
axes[0, 1].set_xlabel('Number of Crossings')
axes[0, 1].set_ylabel('Frequency')

sns.boxplot(x='Border', y='Value', data=data, palette='coolwarm', ax=axes[1, 0])
axes[1, 0].set_title('Distribution of Crossing Values by Border')

plt.tight_layout()
plt.show()

correlation_matrix = data.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='mako', fmt='.2f')
plt.title('Correlation Between Numeric Features')
plt.tight_layout()
plt.show()

top_measures = data.groupby('Measure')['Value'].sum().sort_values(ascending=False)
top5 = top_measures[:5]
others = pd.Series(top_measures[5:].sum(), index=['Other'])
final_measures = pd.concat([top5, others])
colors = ['#00BBF9', '#FEE440', '#9B5DE5', '#00F5D4', '#F15BB5', '#A0C4FF']
final_measures.plot(kind='pie', autopct='%1.1f%%', colors=colors, ylabel='', title='Crossings by Measure Type')
plt.tight_layout()
plt.show()

daily_total = data.groupby('Date')['Value'].sum()
plt.figure(figsize=(12, 6))
daily_total.plot(kind='line', color='#3A86FF', marker='o')
plt.title('Total Crossings Over Time')
plt.xlabel('Date')
plt.ylabel('Total Crossings')
plt.grid(True)
plt.tight_layout()
plt.show()

data['Month'] = data['Date'].dt.month
X = data[['Month']]
y = data['Value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nRegression Model Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))
