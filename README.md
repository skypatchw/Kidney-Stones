# Kidney Stones Prediction (Calcium Oxalate)

Each year millions of people are brought to emergency rooms for kidney stone problems. These stones can be extremely painful and can even cause urinary retention. Determining at-risk patients and pre-emptively initiating therapeutic interventions can prevent the deveolpment of these stones and improve patient outcomes. 

This repository supplies the code of developing a Machine Learning (ML) model that predicts whether an adult, residing in the US, will develop a calcium oxalate kidney stone (1 of the 4 types of kidney stones). The prediction is made based on information about certain biochemical markers.

## Implementation

**ML Model Type**: Supervised

**Algorithm**:

-	Logistic Regression (100% AUC)

**Hyperparameter Tuning Tool**: Grid Search

## Requirements

Python version 3.10.7

Python libraries: pandas, numpy, sklearn, matplotlib, seaborn, joblib

## Data

All necessary data can be found in the "Files" folder of the repository or on Kaggle [www.kaggle.com/datasets/vuppalaadithyasairam/kidney-stone-prediction-based-on-urine-analysis].

## Quick Start

To reproduce the results, download the relevant script and load the corresponding data into a jupyter notebook; running the script will generate all relevant figures and tables. An  executable script file (.py) is also provided if that's more your style. 

## Additional Information

The dataset comprises of urine analyses of 79 patients, collected jointly from a laboratory of the Urology Section, Veteran's Administration Medical Center, Palo Alto, and the Division of Urology, Stanford University School of Medicine. The original dataset was first created in 1985.

It consists of 34 patients who have kidney stones, along with 45 patients who have not developed kidney stones. This information is contained in the column named 'target'. 

## Citation

Andrews, D.F., Herzberg, A.M. (1985). Physical Characteristics of Urines With and Without Crystals. In: Data. Springer Series in Statistics. Springer, New York, NY. https://doi.org/10.1007/978-1-4612-5098-2_45

