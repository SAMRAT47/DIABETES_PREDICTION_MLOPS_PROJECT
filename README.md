🩺 Diabetes Prediction Using Machine Learning


A full-scale ML Ops project that predicts the likelihood of diabetes in individuals based on medical parameters. This project demonstrates the end-to-end lifecycle of a machine learning system — from data ingestion and model training to deployment and continuous integration on cloud infrastructure.

🔍 Project Overview
This application allows users to input medical details like glucose level, insulin, BMI, and other health indicators, and receive a real-time prediction on the likelihood of diabetes.

The goal of this project is to demonstrate:

ML model development and evaluation

Model management using AWS S3

CI/CD with GitHub Actions and Docker

Deployment on AWS EC2

Interactive frontend with Streamlit

🧱 Project Architecture
text
Copy
Edit
               +----------------------+
               |   Streamlit UI App   |
               +----------+-----------+
                          |
                          v
         +-------------------------------+
         |    Prediction Pipeline        |
         | (Preprocessing + Inference)   |
         +-------------------------------+
                          |
                          v
                +-------------------+
                |   Trained Model   |
                |    (from S3)      |
                +-------------------+

    +----------------------------------------------+
    |  Model Training Pipeline (via training.py)   |
    |                                              |
    |  - Data Ingestion from MongoDB               |
    |  - Validation, Transformation, Model Trainer |
    |  - Model Evaluation & S3 Pusher              |
    +----------------------------------------------+
🧠 Model Pipeline Highlights
Data Source: MongoDB Atlas

Validation: Schema-based validation of incoming data

Transformation: Feature scaling, encoding, and preprocessing via sklearn pipeline

Model Training: Uses classification models with performance tracking

Evaluation: Only pushes model to S3 if performance improves

Deployment: Trained model and pipeline deployed via Streamlit on AWS EC2

🚀 Deployment Details
Model Artifact Storage: AWS S3

CI/CD: GitHub Actions configured with Docker and self-hosted runner on EC2

Hosting: Streamlit app deployed on EC2 instance with port open to public

Prediction Endpoint: Direct model inference pipeline integrated into the Streamlit frontend

🌐 Live Application
App URL: https://diabetespredictionmlopsproject-gyupmbyqqvr7exjhmwvyh7.streamlit.app/

🖼️ Screenshot of the deployed application:

![image:](https://github.com/SAMRAT47/Dashboard_Projects/blob/excel_dashboard_branch/Fern%20and%20Petal/fnp%20dashboard.PNG)


🛠️ Tools & Technologies
Python, Scikit-learn, Pandas, Streamlit

MongoDB Atlas (Data Storage)

AWS S3 (Model Storage)

Docker, GitHub Actions (CI/CD)

EC2 (Deployment)

PyYAML, Logging, Custom Exception Handling

📂 Folder Structure
arduino
Copy
Edit
Diabetes_Prediction/
│
├── src/
│   ├── components/
│   ├── config/
│   ├── entity/
│   ├── pipelines/
│   ├── utils/
│   ├── aws_storage/
│
├── templates/
├── static/
├── app.py
├── training.py
├── requirements.txt
├── setup.py
├── Dockerfile
└── .github/workflows/aws.yaml
📌 Features
End-to-end modular codebase following ML Ops best practices

Environment-agnostic configurations for portability

CI/CD-enabled for automatic deployment

Integrated with cloud-native storage and compute

📬 Contact
Feel free to connect with me via LinkedIn or check out more projects at github.com/SAMRAT47


