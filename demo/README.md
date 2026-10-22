# 🩸 HemoVision AI — IDA & Beta Thalassemia Trait Classifier

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![BTT Sensitivity](https://img.shields.io/badge/BTT%20Sensitivity-100.0%25-brightgreen.svg)]()
[![Pure Cohort Accuracy](https://img.shields.io/badge/IDA%2FBTT%20Accuracy-99.7%25-blue.svg)]()

> **Clinical Hemoglobinopathy Decision Support & Interactive Explainable AI (XAI) Cockpit**  
> Differentiating Iron Deficiency Anemia (IDA) from Beta Thalassemia Trait (BTT) using HPLC data, complete hematological indices, and calibrated machine learning models.

---

## 📸 Key Features

- **⚡ Real-Time Diagnostic Inference:** Instant probabilistic classification between **IDA (Iron Deficiency Anemia)** and **BTT (Beta Thalassemia Trait)** powered by `best_logistic_regression_model 2.joblib`.
- **⚖️ Mentzer Index vs. AI Concordance Engine:** Identifies discordant cases (e.g., Mentzer $\ge 13$ indicating IDA despite patient presenting microcytosis with erythrocytosis typical of BTT).
- **🌊 SHAP Waterfall Attribution:** Deconstructs patient feature contributions (RBC Count, MCV, RDW-CV, MCHC, Hemoglobin) pushing probability toward BTT or IDA.
- **🗺️ Interactive 2D Decision Space Surface:** Real-time 2D projection and log-odds boundary with interactive coordinate crosshairs, custom feature axes, and patient trajectory.
- **👥 Nearest Cohort Exemplars:** Identifies nearest historical clinical cases from the 1,190-patient reference HPLC registry (`HPLC_data_cleaned (3).xlsx`).
- **📊 What-If Sensitivity Response Curves:** Real-time simulations showing how marginal variations in RBC Count or RDW-CV alter diagnostic certainty.
- **📑 Formal Clinical PDF Audit Export:** Generates printable, timestamped clinical audit reports ready for laboratory records.

---

## 🚀 Quickstart: Run Code

### Option 1: One-Click Launchers

#### On Windows:
Double-click `run.bat` or run:
```cmd
run.bat
```
*(Automatically creates `.venv`, installs dependencies from `requirements.txt`, launches the server, and opens your browser).*

#### On macOS / Linux:
```bash
chmod +x run.sh
./run.sh
```

---

### Option 2: Standard Terminal Run

#### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/hemovision-ai.git
cd hemovision-ai
```

#### 2. Create and activate a virtual environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
- **Windows (CMD):**
  ```cmd
  python -m venv .venv
  .venv\Scripts\activate.bat
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. Run the Web Application
```bash
python main.py
```
*or directly with the FastAPI server:*
```bash
python server.py
```

#### 5. Open in your browser
Navigate to:  
👉 **[http://localhost:8000](http://localhost:8000)**  
👉 **API Docs (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option 3: Docker Run

```bash
docker build -t hemovision-ai .
docker run -p 8000:8000 hemovision-ai
```
Access at: **http://localhost:8000**

---

## 📂 Project Repository Structure

```text
├── main.py                               # Universal application launcher (auto opens browser)
├── server.py                             # FastAPI application & REST XAI endpoints
├── app.py                                # Shortcut entrypoint
├── requirements.txt                      # Pinned Python package dependencies
├── run.bat                               # 1-click launcher for Windows
├── run.sh                                # 1-click launcher for macOS/Linux
├── Dockerfile                            # Production container configuration
├── .gitignore                            # Git tracking exclusions
├── best_logistic_regression_model 2.joblib # Active trained L2 diagnostic model
├── scaler.joblib                         # Feature standardization pipeline
├── HPLC_data_cleaned (3).xlsx            # Reference 1,190 patient cohort dataset
│
└── static/                               # Frontend Cockpit Assets
    ├── index.html                        # Glassmorphic responsive interface layout
    ├── style.css                         # Desktop cockpit & mobile responsive CSS
    ├── app.js                            # Real-time inference, Canvas & SHAP visualizer
    └── logo.jpg                          # HemoVision AI clinical brand icon
```

---

## ⚖️ Clinical Disclaimer

*HemoVision AI is developed as an investigational Clinical Decision Support and research tool. It provides probabilistic screening and explainability metrics to assist trained hematologists and medical laboratory professionals. All final clinical diagnoses must be confirmed with definitive Hemoglobin Variant HPLC analysis, Serum Ferritin, Hemoglobin Electrophoresis, and DNA genetic testing.*

---

## 📄 License
This project is licensed under the MIT License.
