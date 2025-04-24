# Border Crossing Data Analysis

This project analyzes border crossing data using Python. It includes data cleaning, exploratory data analysis (EDA), and a simple linear regression model to predict crossing values based on the month.

## Features

1. **Data Cleaning**:
   - Handles missing values by filling them with the mode (for categorical data) or mean (for numerical data).
   - Converts the `Date` column to a datetime format.

2. **Exploratory Data Analysis (EDA)**:
   - Visualizes total crossings by border.
   - Displays the distribution of crossing values.
   - Shows the correlation between numeric features.
   - Highlights the top 5 crossing measures using a pie chart.
   - Plots total crossings over time.

3. **Linear Regression Model**:
   - Predicts crossing values based on the month.
   - Evaluates the model using Mean Squared Error (MSE) and R-squared score.

## Requirements

- Python 3.x
- Libraries: pandas, numpy, seaborn, matplotlib, scikit-learn
