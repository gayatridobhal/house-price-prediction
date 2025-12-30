# House Price Prediction – End-to-End ML Pipeline

## About the Project

This project predicts **house sale prices** using regression models built on structured housing data.
It demonstrates a **complete machine learning lifecycle**, including:

* Data preprocessing
* Feature engineering
* Model training & evaluation
* REST API deployment using **FastAPI**
* Containerization with **Docker**
* Testing and project structuring best practices

---

## Dataset Source

* **Kaggle Dataset:**
  **House Prices – Advanced Regression Techniques**
* Contains residential home sales data with **79 explanatory variables**
* Target variable: `SalePrice`

---

## Tech Stack

| Category         | Tools          |
| ---------------- | -------------- |
| Language         | Python         |
| ML               | Scikit-learn   |
| Data             | Pandas, NumPy  |
| API              | FastAPI        |
| Server           | Uvicorn        |
| Testing          | Pytest         |
| Containerization | Docker         |
| CI/CD            | GitHub Actions |

---

## Project Structure

```
house-price-prediction/
│
├── src/            # Core reusable logic (preprocessing, training, API)
├── tests/          # Unit tests
├── model/         # Saved trained models
├── data/           # Raw and processed datasets
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Author

**Gayatri Dobhal**
