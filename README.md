# Student Placement Prediction Project

![Python](https://img.shields.io/badge/Python-3.9-blue.svg) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0-orange.svg) ![Pandas](https://img.shields.io/badge/Pandas-1.3-yellow.svg) ![Seaborn](https://img.shields.io/badge/Seaborn-0.11-green.svg)

A machine learning project to predict student placement outcomes based on academic performance and other personal attributes. This project involves data analysis, feature engineering, and the implementation of classification models to identify the key factors that influence a student's likelihood of getting placed.

---

## Table of Contents
- [Project Goal](#project-goal)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Performance](#model-performance)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)

---

## Project Goal
The primary objective of this project is to build a robust machine learning model that can accurately predict whether a student will be placed in a company. The model aims to provide insights into which factors are most predictive of placement success.

---

## Dataset
The dataset contains various academic and personal attributes of students. The features include:
- `College_ID`: Unique identifier for the college.
- `IQ`: The student's IQ score.
- `Prev_Sem_CGPA`: Cumulative Grade Point Average from the previous semester.
- `CGPA`: Current overall Cumulative Grade Point Average.
- `Academic`: A score representing overall academic performance (out of 10).
- `Internship`: Whether the student has internship experience (Yes/No).
- `Extra_Curri`: A score for extra-curricular involvement (out of 10).
- `Communic`: A score for communication skills (out of 10).
- `Projects_Co`: The number of projects completed.
- `Placement`: The target variable; whether the student was placed (Yes/No).

---

## Methodology
The project follows a standard data science workflow:
1.  **Data Loading & Cleaning:** Loaded the dataset and converted categorical 'Yes'/'No' values into numerical format (1/0).
2.  **Exploratory Data Analysis (EDA):** Used visualization libraries like Matplotlib and Seaborn to explore relationships between features and the placement outcome.
3.  **Feature Preprocessing:** Scaled numerical features using `StandardScaler` to prepare the data for modeling.
4.  **Model Building:** Split the data into training and testing sets and trained two different classification models:
    * Logistic Regression
    * Random Forest Classifier
5.  **Model Evaluation:** Evaluated the models based on Accuracy, AUC Score, and a detailed Classification Report.

---

## Exploratory Data Analysis (EDA)

#### Correlation Matrix
The heatmap shows the correlation between different features. `CGPA` and `Academic` performance show a strong positive correlation with `Placement`.

![Correlation Heatmap](https://github.com/asishghos/Student-Placement/blob/main/images/Figure_2.png)

#### CGPA vs. Placement
The boxplot clearly indicates that students who were placed generally have a higher CGPA than those who were not.

![CGPA vs Placement](https://github.com/asishghos/Student-Placement/blob/main/images/Figure_3.png)

#### Feature Importance
The Random Forest model identified `CGPA`, `Prev_Sem_CGPA`, and `IQ` as the most important features for predicting placements.

![Feature Importance](https://github.com/asishghos/Student-Placement/blob/main/images/Figure_5.png)


---

## Model Performance

The models were evaluated on the test set, and the results are summarized below.

| Model | Accuracy | AUC Score | Key Insight |
| :--- | :---: | :---: | :--- |
| **Logistic Regression** | 90.35% | 0.777 | Good baseline model, but struggles to correctly identify placed students (low recall for class 1). |
| **Random Forest** | **99.9%** | **0.997** | Extremely high performance, suggesting it captures the patterns in the data almost perfectly. |

The Random Forest Classifier significantly outperformed the Logistic Regression model, achieving near-perfect accuracy on the test data.

---

---

## Technologies Used
- **Python 3.9**
- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical operations.
- **Matplotlib & Seaborn:** For data visualization.
- **Scikit-learn:** For machine learning model implementation and evaluation.

---

## How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/student-placement-prediction.git](https://github.com/your-username/student-placement-prediction.git)
    cd student-placement-prediction
    ```
2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You will need to create a `requirements.txt` file by running `pip freeze > requirements.txt` in your terminal)*

4.  **Run the script:**
    ```bash
    python project.py
    ```
