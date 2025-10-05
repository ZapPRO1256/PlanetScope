# Exoplanet Classification Project

This project is designed to **unify different astronomical databases (Kepler, K2, TOI)** and **classify exoplanets** using the **LightGBM** model.
It consists of three main parts:
1. **create_dataset.py** — data unification and unification.
2. **gbm4.py** — training the LightGBM model for classification.
3. **api.py** — REST API for obtaining predictions.

---

## Installation and Startup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/exoplanet-classifier.git
cd exoplanet-classifier
```

### 2. Create a virtual environment
(recommended for dependency isolation)
```bash
python -m venv venv
source venv/bin/activate # Linux / macOS
venv\Scripts\activate # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
exoplanet-classifier/
│
├── create_dataset.py # Unify Kepler / K2 / TOI data into a common CSV
├── gbm4.py # Train a LightGBM model and save it to pkl
├── api.py # FastAPI REST API for classifying new samples
├── requirements.txt # List dependencies
└── README.md # This file
```

---

## Project steps

### 1. Prepare the dataset
All source CSV files (Kepler, K2, TOI) must be placed in the project directory.
Then run:
```bash
python create_dataset.py
```

The script will merge the data into the final file `yea.csv`.

---

### 2. Train the model
After creating the dataset, run:
```bash
python gbm4.py
```

The script:
- performs data preprocessing;
- selects the most correlated features;
- trains the LightGBM model;
- plots feature importance and loss graphs;
- saves the model in `exoplanet_lgb_model.pkl`;
- saves the scaler in `scaler.pkl`.

---

### 3. Launching the API
After training, you can launch the local REST API:
```bash
uvicorn api:app --reload
```

The API will be available at:
```
http://127.0.0.1:8000
```

---

## API Usage

### Endpoint `/predict`
**Method:** `POST`
**Description:** accepts a list of numeric features (`features`) and returns a class prediction.

#### Example query:
```json
{
"features": [365.25, 12.5, 1000.3, 1.3, 0.8, 500, 0.1, 89, 0.02, 1.1, 0.05, 5778, 1.0, 1.0, 4.4, 0.02, 4.6, 0]
}
```

#### Response:
```json
{
"predicted_label": 1,
"confidence": 0.8732
}
```

---

### Endpoint `/`
Check if API is running:
```bash
GET http://127.0.0.1:8000/
```
**Response:**
```json
{ "message": "Exoplanet Classifier API is running!" }
```

---

## Model
The model uses **LightGBM** with parameter settings for class balancing and early stopping control.
The metrics are output to the console after training:
- Accuracy
- ROC AUC
- Classification Report
- Confusion Matrix
- Feature Importance Chart

---

## Example of a complete script

```bash
# 1. Create a combined dataset
python create_dataset.py
# 2. Train the model
python gbm4.py
# 3. Run the REST API
uvicorn api:app --reload
```

## License
This project is distributed under the **MIT** license.
You are free to use and modify it with attribution.
