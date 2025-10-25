
# 🚕 Uber DL Project — Uber Trips Dashboard

This repository (Uber-DL_ProJect) contains a Streamlit dashboard and modelling pipelines for Uber trip analysis. The dashboard (`dashboard.py`) provides interactive EDA, forecasting (Prophet, CNN, LSTM), and ML baselines for trip duration prediction.

---

## 📊 Features

- **Data Filtering** by date and hour
- **Missing Value Handling**
- **Interactive Visualizations** using Plotly
- **Trip Pattern Analysis**
- **Forecasting** using Facebook Prophet
- **Machine Learning Models** to predict trip duration:
  - Linear Regression
  - Decision Tree
  - Random Forest
- **Model Performance Metrics**: RMSE & R² Score comparison
- Clean, responsive UI using **custom HTML + CSS** in Streamlit

---

## Project layout

```
Uber-DL_ProJect/
├── dashboard.py            # Streamlit app (main)
├── Data.csv                # Example dataset (user-provided)
├── requirements.txt        # Python dependencies
├── cnn_model.h5            # (optional) pre-trained model — usually ignored
├── lstm_model.h5           # (optional) pre-trained model — usually ignored
├── assets/                 # images and media for README or app (ignored)
└── README.md               # This file
```

---

## Requirements

Install dependencies from `requirements.txt`.

Windows (cmd.exe):

```cmd
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Linux / macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: `tensorflow` is large. If you only need CPU support, replace `tensorflow` with `tensorflow-cpu` in `requirements.txt`.

---

## Run the dashboard

From the repository root, run:

```cmd
streamlit run dashboard.py
```

Open the URL Streamlit prints in your browser.

---

## 🌐 Live Demo

Access the live app here: [Streamlit Dashboard](uber-trip-analysis-99v4g3r8x4yvupvbjnasoq.streamlit.app)

---

## 📸 Screenshot

![Dashboard Screenshot](assets/image.png)
![Trip features](assets/image1.png)
![Insights](assets/image2.png)
![ML Models](assets/image3.png)
---

## 👨‍💻 Developed By

- Sahil Bhalekar 
- Jash Bheda 
- Om Chavan 

---

## 📝 License

MIT License – feel free to use, modify, and distribute.

---

## 💡 Future Improvements

- Geolocation map-based analysis
- Outlier detection for duration
- Cluster analysis on routes
- Realtime integration with Uber APIs (if available)

---

## Notes & tips

- The default dataset file is `Data.csv`. If your file has a different name, set the CSV path in the Streamlit sidebar when running the app.
- Prophet support is optional. The app will warn if Prophet isn't installed and skip Prophet-related features.
- Large model files (e.g. `*.h5`, `*.pkl`) and the `assets/` folder are ignored by `.gitignore` to avoid accidentally committing heavy binaries. Use Git LFS if you need to track large models.

