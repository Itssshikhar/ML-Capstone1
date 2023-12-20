# ML-Capstone1-Project

### Machine Leaning Model for Heart Disease Prediction 

Welcome to our Machine Learning repository dedicated to Heart Disease prediction. This collection features a diverse dataset and explores predictive models to enhance understanding and accuracy in identifying potential cardiovascular risks.

Github repo: https://github.com/Itssshikhar/ML-Capstone1
Dataset: https://www.kaggle.com/datasets/utkarshx27/heart-disease-diagnosis-dataset

### Dataset

Heart Disease Prediction Dataset. Predicting the Presence or Absence of Heart Disease Based on Various Factors

#### Features:

  -- 1. age       
  -- 2. sex       
  -- 3. chest pain type  (4 values)       
  -- 4. resting blood pressure  
  -- 5. serum cholestoral in mg/dl      
  -- 6. fasting blood sugar > 120 mg/dl       
  -- 7. resting electrocardiographic results  (values 0,1,2) 
  -- 8. maximum heart rate achieved  
  -- 9. exercise induced angina    
  -- 10. oldpeak = ST depression induced by exercise relative to rest   
  -- 11. the slope of the peak exercise ST segment     
  -- 12. number of major vessels (0-3) colored by flourosopy        
  -- 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect
  -- 14. Target(Absence (1) or presence (2) of heart disease) 

## Capstone1 Project Requirements (Evaluation Criteria)

- Problem description
- EDA
- Model training
- Exporting notebook to script
- Model deployment
- Reproducibility
- Dependency and environment management
- Containerization
- Cloud deployment

### Dependency and Environment Management Guide

You can easily install dependencies from requirements.txt and use virtual environment.

- `pip install pipenv`

- `pip shell`

- `pip install -r requirements.txt`

If can't or don't know how to, here are the needed packages, just run

- `pip install pipenv Flask==3.0.0
graphviz==0.20.1
matplotlib==3.8.0
numpy==1.26.1
pandas==2.1.2
Requests==2.31.0
scikit_learn==1.3.1
seaborn==0.13.0`

### Depolyment Guide

#### To run it locally:

- Run `python predict.py` on a terminal
- Open a terminal and run python `test_predict.py`

#### To run it docker:

- Download and run Docker Desktop: https://www.docker.com/

- Open a terminal

- `docker build -t capstone_test .`

- `docker run -it --rm -p 6969:6969 capstone_test`

- Open a new terminal and run python `test_predict.py`

#### To run it in cloud:

- This is still being worked on.
