Here are both versions for your advertising sales prediction project.



**Advertising Sales Prediction with Linear Regression**
Built a predictive analytics model to forecast sales based on advertising budget allocation across TV, Radio, and Newspaper channels. Performed comprehensive data analysis including correlation heatmaps and feature standardization, achieving a model with **RMSE of 1.56**, identifying TV advertising as the strongest predictor (correlation: 0.78). The project demonstrates end-to-end machine learning workflow from EDA to model evaluation.

**Tags:** `Linear Regression` `Predictive Analytics` `Feature Correlation` `Advertising Analytics` `Scikit-learn`

---



```markdown
# 📈 Advertising Sales Prediction with Linear Regression



A machine learning project that builds a **Linear Regression model** to predict product sales based on advertising budget allocation across different media channels (TV, Radio, Newspaper). This project demonstrates a complete data science workflow from exploratory data analysis to model evaluation.

## 🎯 Project Objective

To develop a predictive model that helps businesses:
- Understand which advertising channels most significantly impact sales
- Forecast sales based on planned advertising budgets
- Optimize marketing budget allocation for maximum ROI

## 📊 Dataset

**Source:** Kaggle Advertising Dataset  
**Features:** 200 observations with 4 variables
- `TV`: Advertising budget spent on TV (in thousands of dollars)
- `Radio`: Advertising budget spent on Radio (in thousands of dollars)
- `Newspaper`: Advertising budget spent on Newspaper (in thousands of dollars)
- `Sales`: Product sales (in thousands of units) - **Target Variable**

## 🔬 Methodology

### 1. **Exploratory Data Analysis (EDA)**
- Initial data inspection and descriptive statistics
- Correlation analysis between features and target variable
- Visualization using scatter plots and heatmaps

### 2. **Key Finding**
- **TV advertising** showed the strongest positive correlation with sales (0.78)
- Radio advertising had moderate correlation (0.58)
- Newspaper advertising showed weakest correlation (0.23)

![Correlation Heatmap](https://via.placeholder.com/600x400/2C3E50/FFFFFF?text=Correlation+Heatmap+Here)
*Heatmap showing correlation between advertising channels and sales*

### 3. **Data Preprocessing**
- Removed unnecessary columns
- Feature-target separation: `X` (TV, Radio, Newspaper) and `y` (Sales)
- Standardization using `StandardScaler` for consistent feature scaling
- Train-test split (80-20 ratio)

### 4. **Model Development**
- Implemented **Linear Regression** using scikit-learn
- Trained on standardized features
- Made predictions on test set

### 5. **Model Evaluation**
| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Mean Absolute Error (MAE)** | 1.14 | Average prediction error in sales units |
| **Mean Squared Error (MSE)** | 2.43 | Penalizes larger errors more heavily |
| **Root Mean Squared Error (RMSE)** | 1.56 | In same units as target variable |

## 🛠️ Technology Stack

```python
# Core Libraries
import numpy as np          # Numerical computations
import pandas as pd         # Data manipulation
import seaborn as sns       # Statistical visualization
import matplotlib.pyplot as plt # Plotting

# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
```

## 📁 Project Structure

```
advertising-sales-prediction/
├── data/
│   └── advertising.csv          # Original dataset
├── notebooks/
│   └── advertising_analysis.ipynb  # Jupyter notebook with complete analysis
├── src/
│   ├── data_preprocessing.py    # Data cleaning functions
│   ├── model_training.py        # Model training pipeline
│   └── visualization.py         # Plotting functions
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── LICENSE                      # MIT License
```

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/advertising-sales-prediction.git
cd advertising-sales-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```python
# Load and preprocess data
from src.data_preprocessing import load_and_prepare_data
X_train, X_test, y_train, y_test = load_and_prepare_data('data/advertising.csv')

# Train model
from src.model_training import train_linear_regression
model, scaler = train_linear_regression(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

## 📈 Results & Insights

### Model Performance
The Linear Regression model achieved **RMSE of 1.56**, meaning predictions are typically within 1,560 units of actual sales. Given sales range from ~5,000 to 25,000 units, this represents approximately **6-31% relative error**.

### Business Implications
1. **TV advertising** provides the best return on investment
2. **Radio advertising** offers moderate effectiveness
3. **Newspaper advertising** has limited impact on sales
4. Businesses should prioritize TV and Radio channels for maximum sales impact

## 🔮 Future Enhancements

- [ ] **Multiple Regression Models**: Compare with Ridge, Lasso, and Polynomial Regression
- [ ] **Interaction Terms**: Investigate synergy between advertising channels
- [ ] **Time Series Analysis**: Incorporate temporal effects in advertising
- [ ] **Deployment**: Create a web interface for budget planning
- [ ] **A/B Testing Framework**: Simulate different budget allocation strategies

## 📚 Learning Outcomes

This project reinforced:
- **Feature correlation analysis** and interpretation
- **Data standardization** importance in regression models
- **Model evaluation metrics** for regression tasks
- **Business interpretation** of machine learning results

## 👥 Author

**Mohawiz Hamid**  
- GitHub: [@mohawiz](https://github.com/mohawiz)




## 🙏 Acknowledgements

- Kaggle for providing the Advertising Dataset
- Scikit-learn documentation and community


---

**📊 Model Summary:** Simple yet effective Linear Regression model demonstrating strong predictive power for sales based on advertising budgets.
```

## 💡 **How to Use These Files**

1. **For your portfolio website**: Use the **Brief Portfolio Description**
2. **For GitHub repository**: Use the complete **README.md** file
3. **For LinkedIn/Resume**: Use a condensed version:

> "Developed a Linear Regression model predicting sales from advertising budgets, identifying TV ads as the strongest predictor (r=0.78). Achieved RMSE of 1.56 through comprehensive EDA, feature standardization, and model evaluation."

Would you like me to help you create the actual Jupyter notebook or Python scripts for this project as well?
