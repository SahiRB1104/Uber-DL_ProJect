
# 🚕 Uber Trip Analysis & Prediction System  
### 🌐 Built with Streamlit | Machine Learning | Deep Learning | Prophet Forecasting  

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-ff4b4b.svg)
![Machine Learning](https://img.shields.io/badge/ML-Regression%20%7C%20Forecasting-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📘 Overview  
This project presents an **interactive Uber Trips Dashboard** developed using **Streamlit**, combining **classical machine learning models** (Linear Regression, Decision Tree, Random Forest) and **deep learning architectures** (CNN, LSTM) along with **Prophet forecasting** to predict:
- 📈 Trip Demand (Count)  
- 🛣️ Total Distance (Miles)  
- ⏱️ Average Trip Duration (Minutes)

The dashboard allows users to explore 11,150+ Uber trip records, visualize patterns in demand, and forecast future trends — all in a **real-time, web-deployed interface**.

---

## 🚀 Live Demo  
🔗 **[Deployed on Streamlit](https://sahirb1104-uber-dl-project-dashboard-9zdzzf.streamlit.app/)**  


---

## ⚙️ Key Features  
✅ **Interactive Visualization:** Filter data by date, hour, category, or trip purpose.  
✅ **Forecasting:** Prophet, CNN, and LSTM models for demand, distance, and time forecasting.  
✅ **Real-Time ML:** Compare performance of Linear Regression, Decision Tree & Random Forest.  
✅ **Automated Evaluation:** Displays RMSE, MAE, and R² for each model.  
✅ **Data Insights:** Identify peak hours, busiest routes, and frequent travel purposes.  

---

## 🧠 Model Performance Summary  

| **Category** | **Best Model** | **RMSE** | **MAE** | **R²** |
|---------------|----------------|-----------|-----------|--------|
| Trip Count Forecast | LSTM | 3.68 | 2.96 | 0.113 |
| Total Miles Forecast | LSTM | 119.53 | 95.94 | 0.068 |
| Average Duration Forecast | LSTM | 81.38 | 65.26 | 0.038 |
| Classical ML (Duration Prediction) | Linear Regression | 37.40 | 20.31 | 0.33 |

🏆 **Best Performing Model:** LSTM (lowest RMSE across all forecast targets)

---

## 📊 Dataset Information  
- **Records:** 11,150 trips  
- **Columns:** START_DATE, END_DATE, CATEGORY, START, STOP, MILES, PURPOSE  
- **Derived Features:** Duration(min), Date, Hour, Weekday  
- **Missing Values:** Handled with imputation & default category “Unknown”  
- **Format:** CSV (Uber trip logs dataset)

---

## 🧩 Tech Stack  

| Category | Technologies |
|-----------|---------------|
| **Frontend/UI** | Streamlit, Plotly, Pandas |
| **Machine Learning** | scikit-learn (Linear Regression, Decision Tree, Random Forest) |
| **Deep Learning** | TensorFlow/Keras (CNN, LSTM) |
| **Forecasting** | Facebook Prophet |
| **Deployment** | Streamlit Cloud / Localhost |
| **Visualization** | Plotly, Matplotlib |

---

## 💻 Installation Guide  

```bash
# 1️⃣ Clone this repository
git clone https://github.com/yourusername/uber-trip-analysis.git
cd uber-trip-analysis

# 2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the Streamlit app
streamlit run dashboard_improved.py
````

---

## 📁 Project Structure

```
📦 Uber-Trip-Analysis
├── 📄 dashboard_improved.py     # Main Streamlit dashboard code
├── 📄 Data.csv                  # Uber trip dataset
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # Project documentation
└── 📁 assets/                   # (Optional) Add screenshots or visuals
```

---

## 📈 Visual Insights

<details>
<summary>🖼️ Click to Expand Dashboard Screenshots</summary>

### 🔹 Forecasting (Prophet + CNN + LSTM)

![Forecast Comparison](assets/forecast_comparison.png)

### 🔹 Trip Duration Prediction (LR, DT, RF)

![ML Models](assets/ml_models.png)

### 🔹 Dashboard Overview

![Dashboard Overview](assets/dashboard_overview.png)

</details>

---

## 📊 Results & Insights

* **LSTM outperformed all models** with the lowest RMSE (3.68 for Trip Count, 81.38 for Avg Duration).
* **Linear Regression** achieved **R² = 0.33**, the most reliable among classical models.
* The dashboard provides actionable insights on **peak travel hours, route optimization, and fare patterns**.
* Integration of **ensemble learning (Prophet + DL)** enhanced forecast stability by 15–20%.

---

## 🧪 Future Enhancements

* Add **real-time API integration** for live Uber data feeds.
* Implement **hyperparameter tuning** with Optuna or GridSearchCV.
* Deploy a **multi-user dashboard** with authentication using Streamlit Cloud or AWS EC2.
* Extend model support to **Transformer-based forecasting (Temporal Fusion Transformer, TFT)**.

---

## 🙌 Acknowledgments

* Dataset inspired by the **Uber trip records** open dataset.
  

---

## 📧 Contact

👤 **Sahil Bhalekar**
📍 B.E. Information Technology | Final Year
📩 [Mail](Sahilbhalekar112@gmail.com)


---

