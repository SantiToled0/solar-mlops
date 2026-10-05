
# Solar MLOps

End-to-end machine learning project for photovoltaic power prediction, combining data analysis, machine learning, forecasting and MLOps practices.

The project is based on photovoltaic data analysis work developed during my internship at ITER. This repository uses a different photovoltaic dataset to build a public and reproducible version of the workflow.

## Overview

The main goal is to predict **AC power generation** from photovoltaic system measurements.

The project covers the full machine learning workflow:

**Data processing → Exploratory Data Analysis → Feature Engineering → Model Training → Model Evaluation → Forecasting → API / MLOps → Docker**

The project is also designed as a practical example of how a machine learning model can be integrated into a more complete and reproducible software workflow.

---

## Project Workflow

### 1. Data Preparation

The original photovoltaic data is cleaned and transformed into a structured dataset suitable for machine learning.

The preprocessing workflow includes:

* Data cleaning and validation
* Timestamp processing
* Handling of missing or inconsistent values
* Feature generation
* Saving the processed dataset in Parquet format

The main target variable is:

`ac_power`

Examples of input features include:

* Irradiance
* Ambient temperature
* Module temperature
* Time-related features
* Seasonal/cyclical features

---

### 2. Exploratory Data Analysis

The exploratory analysis is used to understand the behaviour of the photovoltaic system and the relationship between the variables.

The analysis includes:

* Distribution analysis
* Time series exploration
* Correlations between variables
* Daily and seasonal patterns
* Relationship between environmental conditions and AC power

Feature engineering is also performed during this stage to represent cyclical temporal information more effectively.

---

### 3. Machine Learning Models

Several regression models are trained and compared in order to determine which approach performs best for the problem.

The project includes:

* Linear Regression
* Random Forest Regressor
* HistGradientBoosting Regressor
* XGBoost Regressor
* Neural Network implemented with TensorFlow

The models are evaluated using:

* **MAE** — Mean Absolute Error
* **RMSE** — Root Mean Squared Error
* **R²** — Coefficient of Determination

### Example results

| Model                |   MAE     |      RMSE |         R² |
| -------------------- | ---------:| --------: | ---------: |
| Random Forest        | **10.13** |     31.56 |     0.9871 |
| HistGradientBoosting | 11.79     |     35.20 |     0.9839 |
| Neural Network       | 11.78     | **28.74** | **0.9893** |

The neural network achieved the best RMSE and R² among the models shown above, while Random Forest obtained the lowest MAE.

---

## 4. Forecasting

To simulate a real forecasting scenario, the project generates synthetic future weather variables from historical photovoltaic observations.

The forecasting process introduces realistic uncertainty into:

* Irradiance
* Ambient temperature
* Module temperature

The uncertainty increases with the forecasting horizon, making the simulated predictions less certain further into the future.

A rolling multi-day forecast is then generated and used as input to the trained models.

The predicted AC power can subsequently be compared with the real observed values in order to evaluate how the models behave when working with forecasted rather than measured input data.

---

## 5. MLOps

The project extends beyond model development and includes MLOps practices.

### MLflow

MLflow is used to track machine learning experiments and compare model runs.

This allows the workflow to keep track of:

* Model parameters
* Evaluation metrics
* Experiments
* Model artifacts

This makes model experimentation more reproducible and easier to compare.

### PostgreSQL

PostgreSQL is integrated into the project infrastructure to provide a persistent relational database.

### FastAPI

FastAPI provides an API layer for interacting with the machine learning workflow and exposing model functionality through a web service.

### Docker

Docker is used to containerize the project and simplify the setup of its different components.

The environment can be managed using Docker Compose, allowing the main services to run together in a reproducible local environment.

---

## Technology Stack

### Data Science & Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* TensorFlow
* MLflow

### Data & Infrastructure

* PostgreSQL
* FastAPI
* Docker
* Docker Compose

### Development

* Jupyter Notebook
* Git / GitHub
* Linux

---

## Project Structure

At a high level, the project is organised around the following components:

solar-mlops/
│
├── data/
│   ├── processed/
│   │   └── solar_clean.parquet
│   └── ...
│
├── notebooks/
│   ├── exploratory analysis
│   ├── feature engineering
│   ├── model development
│   └── forecasting
│
├── ...
│
├── compose.yaml
├── .env.example
├── .gitignore
└── README.md


The notebooks contain the main stages of the data analysis and model development process, while the containerized environment provides the infrastructure required for the MLOps components.

---

## Running the Project

### Prerequisites

Make sure the following are installed:

* Docker
* Docker Compose
* Git

Clone the repository:

```bash
git clone https://github.com/SantiToled0/solar-mlops.git
cd solar-mlops
```

Create the environment configuration from the example file:

```bash
cp .env.example .env
```

Then start the project with Docker Compose:

```bash
docker compose up -d --build
```

The exact services and configuration depend on the current Docker Compose setup.

---

## Reproducibility

One of the main goals of the project is to make the machine learning workflow reproducible.

Docker is used to provide a consistent environment, while MLflow keeps track of model experiments and results.

This separates the different stages of the workflow and makes it easier to reproduce experiments and move from model development towards deployment.

---

## What This Project Demonstrates

This project combines several aspects of a real machine learning workflow rather than focusing only on model training.

It demonstrates experience with:

* Processing and analysing real-world time series data
* Feature engineering
* Regression modelling
* Comparing different machine learning approaches
* Neural networks with TensorFlow
* Time-series forecasting simulation
* Experiment tracking with MLflow
* Database integration with PostgreSQL
* API development with FastAPI
* Containerization with Docker
* Building an end-to-end ML workflow

---

## Limitations and Future Improvements

The forecasting component is currently a **simulated forecasting process**, rather than a production weather forecasting system.

Possible future improvements include:

* Using external weather forecast data
* Improving the forecasting methodology
* Automated model retraining
* Model monitoring and drift detection
* CI/CD integration
* Cloud deployment
* More advanced time-series models

---

## Motivation

The project is based on photovoltaic data analysis and was developed as a practical application of machine learning and MLOps techniques.

The objective is not only to obtain accurate predictions, but to demonstrate how a machine learning solution can be developed as a complete workflow, from data preparation and experimentation to reproducible deployment.
