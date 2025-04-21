Laptop Price Predictor

A machine learning-powered web application that predicts laptop prices based on key specifications such as brand, RAM, processor type, screen resolution, and more. Built with Python, trained on real-world data, and deployed using Streamlit.

Features:

Predicts laptop prices based on user input

Interactive UI using Streamlit

End-to-end pipeline with data preprocessing, model training, and deployment

Supports features like:

Brand and type

RAM size

Touchscreen and IPS display

Screen size and resolution

Processor and storage details

GPU brand and operating system

Tech Stack:

Python

Pandas, NumPy, Scikit-learn

Jupyter Notebook

Streamlit

Pickle (for model serialization)

Machine Learning Workflow:

Data Cleaning: Handled missing values and standardized formats

Feature Engineering: Extracted and encoded relevant features

Model Training: Trained a regression model to predict laptop prices

Deployment: Integrated the model with a Streamlit frontend for real-time prediction

Key Files:

LaptopPricePredictor.ipynb: Jupyter Notebook containing the full ML pipeline

app.py: Streamlit app for user interaction

pipe.pkl: Serialized model pipeline for prediction

df.pkl: Processed dataset used in the app

How to Run:

Clone the repository

Install required packages

Run the Streamlit app using: streamlit run app.py

Demo:

Input your laptop’s specifications and get an instant estimated price.

Contributions:

Open to feedback and contributions. Feel free to fork the repo and improve it!
