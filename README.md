# fruit-store-ai-ml-sql-powerbi
# 🍎 Fruit Store Intelligence — AI/ML Sales Prediction & Analytics

<p align="center">
  <strong>A Data Science Project Combining Python, Machine Learning, SQL, and Power BI</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python" />
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?logo=scikitlearn" alt="Machine Learning" />
  <img src="https://img.shields.io/badge/SQL-Database-blue?logo=mysql" alt="SQL" />
  <img src="https://img.shields.io/badge/Power%20BI-Analytics-yellow?logo=powerbi" alt="Power BI" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit" alt="Streamlit" />
  <img src="https://img.shields.io/badge/Status-Learning%20Project-success" alt="Project Status" />
</p>

## 📌 Project Overview

**Fruit Store Intelligence** is an end-to-end data science project designed to analyze fruit store sales, monitor business performance, and predict the quantity of fruit that may be sold.

The project combines data analysis, machine learning, SQL-based reporting, and Power BI visualization to transform sales data into actionable business insights.

A Python-powered interactive dashboard allows users to explore revenue, sales quantities, store performance, promotional trends, and machine learning predictions.

## 🎯 Project Objectives

- Analyze fruit store sales and revenue.
- Identify top-performing fruits and stores.
- Understand monthly sales trends.
- Evaluate sales patterns associated with promotions.
- Predict fruit sales quantities using machine learning.
- Prepare sales data for SQL analysis and Power BI reporting.
- Build an interactive dashboard for business decision-making.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data processing and application development |
| Pandas | Data cleaning and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning model development |
| Random Forest Regressor | Sales quantity prediction |
| Joblib | Saving and loading the trained model |
| Streamlit | Interactive web dashboard |
| Plotly | Interactive data visualizations |
| SQL / MySQL | Database storage and business queries |
| Power BI | Business intelligence dashboard |
| Git & GitHub | Version control and project hosting |

## ✨ Key Features

### 📊 Sales Analytics
- Calculate total sales revenue and units sold.
- Compare fruit-wise sales performance.
- Analyze monthly revenue trends.
- Explore revenue by store.
- Filter results by fruit and store.

### 🤖 Machine Learning
- Train a Random Forest regression model.
- Predict expected units sold.
- Use fruit, category, price, promotion, month, and store as model inputs.
- Evaluate model performance using MAE, RMSE, and R².

### 📈 Interactive Dashboard
- KPI cards for business performance.
- Interactive charts using Plotly.
- Fruit and store filters.
- Sales data table.
- Download filtered sales data as a CSV file.
- Prediction form for estimating sales quantities.

### 🗄️ SQL Analytics
Use SQL to query sales data, calculate revenue, compare fruit performance, and analyze store-level results.

### 📉 Power BI Reporting
Create a business intelligence dashboard featuring revenue KPIs, monthly trends, fruit comparisons, store analysis, and interactive slicers.

## 📁 Project Structure

```text
Fruit_Store_AI_ML_SQL_PowerBI_Project/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── fruit_store_sales.csv
│
├── models/
│   └── units_sold_model.joblib
│
└── reports/
    └── model_metrics.txt
```

**Note:** The `models/` and `reports/` output files are generated when the training script runs successfully. The trained model is excluded from GitHub by the example `.gitignore`.

## 📂 Dataset Description

The project uses a sample fruit store sales dataset.

| Column | Description |
|---|---|
| `date` | Date of the sales record |
| `fruit` | Name of the fruit |
| `category` | Fruit category |
| `unit_price` | Selling price per unit |
| `units_sold` | Quantity sold |
| `promotion` | Promotion indicator: 0 or 1 |
| `store` | Store location |

**Important:** The included sample data is intended for demonstration and learning. It should not be treated as actual business sales data.

## ⚙️ Installation and Setup — Windows & VS Code

### Step 1: Install the prerequisites

Install the following software:

- Python 3.11
- Visual Studio Code
- Git
- Power BI Desktop
- MySQL Server and MySQL Workbench, if using MySQL

### Step 2: Open the project

Open the project folder in VS Code.

Open the integrated terminal using **Terminal → New Terminal**.

### Step 3: Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

Activate the environment:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, open Command Prompt in VS Code and run:

```cmd
.venv\Scripts\activate.bat
```

### Step 4: Install Python libraries

Make sure `requirements.txt` contains:

```text
pandas
numpy
scikit-learn
