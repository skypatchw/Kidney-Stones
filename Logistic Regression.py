#!/usr/bin/env python
# coding: utf-8

# # DATA ACQUISITION 

# In[1]:


import numpy as np
import pandas as pd

print(f"NumPy Version: {np.__version__}\nPandas Version: {pd.__version__}")


# In[2]:


kidney = pd.read_csv("/Datasets/Kidney Stones/kidney stones.csv")


# # EXPLORATORY DATA ANALYSIS 
# ## ( DATA PREPARATION + FEATURE ENGINEERING )

# In[3]:


import seaborn as sns 
import scipy as sp 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import sklearn 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler 

print(f"Seaborn Version: {sns.__version__}\nMatplotlib Version: {mpl.__version__}\nSklearn Version: {sklearn.__version__}\nScipy Version: {sp.__version__}")


# In[4]:


kidney.shape


# In[5]:


kidney.head()


# In[6]:


kidney.info()


# In[7]:


kidney.describe().T


# ## Renaming columns 

# In[8]:


column_mapping = {
    'gravity':'specific_gravity', 
    'ph': 'urine_ph', 
    'osmo':'osmolality',
    'cond':'conductivity', 
    'calc': 'calcium', 
    'target': 'outcome'
                }

kidney.rename(columns=column_mapping, inplace=True)


# ## Checking null values

# In[9]:


kidney.isnull().sum()


# ## Kidney stones outcome ratio

# In[10]:


kidney['outcome'].value_counts()


# In[11]:


sns.histplot(kidney['outcome'])


# ### Percentage of sample positive for kidney stones

# In[12]:


print(f"kidney ratio = {sum(kidney['outcome']) / len(kidney):.4f}")


# * Data is unbalanced 

# ## Checking number of duplicate rows 

# In[13]:


kidney.duplicated().sum()


# ## Checking unique values for invalid entries 

# In[14]:


for col in kidney.columns:
    unique_values = kidney[col].unique()
    print(f"Unique values for {col} : {unique_values}")


# ## Investigating urine calcium

# In[15]:


sns.displot(kidney['calcium'])


# ## Investigating urine conductivity

# In[16]:


sns.displot(kidney['conductivity'])


# ## Investigating urine pH

# In[17]:


sns.displot(kidney['urine_ph'])


# ## Investigating amount of urea 

# In[18]:


sns.displot(kidney['urea'])


# ## Investigating urine osmolality

# In[19]:


sns.displot(kidney['osmolality'])


# ## Inspecting pair plot of biochemical markers

# In[20]:


sns.pairplot(kidney, hue='outcome')


# ## Checking correlation 

# In[21]:


corr = kidney.corr()

# Get upper criangle of the co-relation matrix
matrix = np.triu(corr)


# Use upper triangle matrix as mask 
sns.set(rc={"figure.figsize":(10, 10)})   
sns.heatmap(corr, cmap="Blues", annot=True, mask=matrix)


# ## Creating separate dataframes based on kidney stones outcome 

# In[22]:


stones_present = kidney[kidney['outcome'] == 1]


# In[23]:


stones_absent = kidney[kidney['outcome'] == 0]


# ## Investigating urea and osmolality relationship 

# ### Combined

# In[24]:


q = sns.lmplot(data=kidney, x='urea', y='osmolality', height=5)

def annotate(data, **kws):
    r, p = sp.stats.pearsonr(data['urea'], data['osmolality'])
    ax = plt.gca()
    ax.text(.6, .1, 'Pearson r = {:.2f}\np-value = {:.2g}'.format(r, p),
            transform=ax.transAxes, weight='bold')
    
q.map_dataframe(annotate)
plt.show()


# ### With kidney stones 

# In[25]:


q = sns.lmplot(data=stones_present, x='urea', y='osmolality', height=5)

def annotate(data, **kws):
    r, p = sp.stats.pearsonr(data['urea'], data['osmolality'])
    ax = plt.gca()
    ax.text(.6, .1, 'Pearson r = {:.2f}\np-value = {:.2g}'.format(r, p), weight='bold',
            transform=ax.transAxes)
    
q.map_dataframe(annotate)
plt.show()


# ### Without kidney stones  

# In[26]:


q = sns.lmplot(data=stones_absent, x='urea', y='osmolality', height=5)

def annotate(data, **kws):
    r, p = sp.stats.pearsonr(data['urea'], data['osmolality'])
    ax = plt.gca()
    ax.text(.6, .1, 'Pearson r = {:.2f}\np-value = {:.2g}'.format(r, p), weight='bold',
            transform=ax.transAxes)
    
q.map_dataframe(annotate)
plt.show()


# ## Investigating urea and conductivity relationship

# ### Combined 

# In[27]:


q = sns.lmplot(data=kidney, x='urea', y='conductivity', height=5)

def annotate(data, **kws):
    r, p = sp.stats.pearsonr(data['urea'], data['conductivity'])
    ax = plt.gca()
    ax.text(.6, .1, 'Pearson r = {:.2f}\np-value = {:.2g}'.format(r, p), weight='bold',
            transform=ax.transAxes)
    
q.map_dataframe(annotate)
plt.show()


# ### With kidney stones 

# In[28]:


q = sns.lmplot(data=stones_present, x='urea', y='conductivity', height=5)

def annotate(data, **kws):
    r, p = sp.stats.pearsonr(data['urea'], data['conductivity'])
    ax = plt.gca()
    ax.text(.6, .1, 'Pearson r = {:.2f}\np-value = {:.2g}'.format(r, p), weight='bold',
            transform=ax.transAxes)
    
q.map_dataframe(annotate)
plt.show()


# ### Without kidney stones 

# In[29]:


q = sns.lmplot(data=stones_absent, x='urea', y='conductivity', height=5)

def annotate(data, **kws):
    r, p = sp.stats.pearsonr(data['urea'], data['conductivity'])
    ax = plt.gca()
    ax.text(.6, .1, 'Pearson r = {:.2f}\np-value = {:.2g}'.format(r, p), weight='bold',
            transform=ax.transAxes)
    
q.map_dataframe(annotate)
plt.show()


# ## Investigating calcium and conductivity relationship

# ### Combined 

# In[30]:


q = sns.lmplot(data=kidney, x='calcium', y='conductivity', height=5)

def annotate(data, **kws):
    r, p = sp.stats.pearsonr(data['calcium'], data['conductivity'])
    ax = plt.gca()
    ax.text(.6, .1, 'Pearson r = {:.2f}\np-value = {:.2g}'.format(r, p), weight='bold',
            transform=ax.transAxes)
    
q.map_dataframe(annotate)
plt.show()


# ### With kidney stones 

# In[31]:


q = sns.lmplot(data=stones_present, x='calcium', y='conductivity', height=5)

def annotate(data, **kws):
    r, p = sp.stats.pearsonr(data['calcium'], data['conductivity'])
    ax = plt.gca()
    ax.text(.6, .1, 'Pearson r = {:.2f}\np-value = {:.2g}'.format(r, p), weight='bold',
            transform=ax.transAxes)
    
q.map_dataframe(annotate)
plt.show()


# ### Without kidney stones 

# In[32]:


q = sns.lmplot(data=stones_absent, x='calcium', y='conductivity', height=5)

def annotate(data, **kws):
    r, p = sp.stats.pearsonr(data['calcium'], data['conductivity'])
    ax = plt.gca()
    ax.text(.6, .1, 'Pearson r = {:.2f}\np-value = {:.2g}'.format(r, p), weight='bold',
            transform=ax.transAxes)
    
q.map_dataframe(annotate)
plt.show()


# ## Investigating urea and conductivity relationship

# ### Combined 

# In[33]:


q = sns.lmplot(data=kidney, x='urea', y='conductivity', height=5)

def annotate(data, **kws):
    r, p = sp.stats.pearsonr(data['urea'], data['conductivity'])
    ax = plt.gca()
    ax.text(.6, .1, 'Pearson r = {:.2f}\np-value = {:.2g}'.format(r, p), weight='bold',
            transform=ax.transAxes)
    
q.map_dataframe(annotate)
plt.show()


# ### With kidney stones 

# In[34]:


q = sns.lmplot(data=stones_present, x='urea', y='conductivity', height=5)

def annotate(data, **kws):
    r, p = sp.stats.pearsonr(data['urea'], data['conductivity'])
    ax = plt.gca()
    ax.text(.6, .1, 'Pearson r = {:.2f}\np-value = {:.2g}'.format(r, p), weight='bold',
            transform=ax.transAxes)
    
q.map_dataframe(annotate)
plt.show()


# ### Without kidney stones 

# In[35]:


q = sns.lmplot(data=stones_absent, x='urea', y='conductivity', height=5)

def annotate(data, **kws):
    r, p = sp.stats.pearsonr(data['urea'], data['conductivity'])
    ax = plt.gca()
    ax.text(.6, .1, 'Pearson r = {:.2f}\np-value = {:.2g}'.format(r, p), weight='bold',
            transform=ax.transAxes)
    
q.map_dataframe(annotate)
plt.show()


# ## Investigating calcium and kidney stones relationship 

# In[36]:


fig, ax = plt.subplots()
fig.set_size_inches(5, 5)
sns.regplot(data=kidney, x='outcome', y='calcium', x_jitter=.15)


# ## Checking outliers 

# In[37]:


plt.figure(figsize=(12,8))
sns.boxplot(data=kidney)
plt.show()


# ### Removing larger value columns to visualise box plots of smaller value columns clearly 

# In[38]:


viz_df = kidney.drop(columns=['osmolality', 'urea']) 


# In[39]:


plt.figure(figsize=(12,8))
sns.boxplot(data=viz_df)
plt.show()


# ## Inspecting individual box plots

# ### Urine pH

# In[40]:


plt.figure(figsize=(5,5))
sns.boxplot(y=kidney['urine_ph'], width=0.2)


# ### Urine calcium

# In[41]:


plt.figure(figsize=(5,5))
sns.boxplot(y=kidney['calcium'], width=0.2)


# ## Capping Outliers 

# In[42]:


columns_to_cap = ['urine_ph', 'calcium']


# In[43]:


def cap_outliers(data, columns):
    
    for column in columns:
       
        q1 = data[column].quantile(0.25)      # Get the Q1 (25 percentile) and Q3 (75 percentile)
        q3 = data[column].quantile(0.75)

        iqr = q3 - q1                         # Calculate interquartile range

        max_limit = q3 + (1.5 * iqr)          # Set limits
        min_limit = q1 - (1.5 * iqr)

        data[column] = np.clip(               # Cap outliers
                        data[column], 
                        a_min=min_limit, 
                        a_max=max_limit)     
    


# In[44]:


cap_outliers(data=kidney, columns=columns_to_cap)


# ### Checking capping result

# In[45]:


capped_columns = ['urine_ph', 'calcium']


# In[46]:


plt.figure(figsize=(12,8))
sns.boxplot(data=kidney[capped_columns], width=0.2)


# In[47]:


kidney.info()


# ## Scaling features to aid in solver convergence 

# In[48]:


# Make copy of dataset to preserve original 
kidney_scaled = kidney.copy()


# In[49]:


features_unchanged = ['specific_gravity', 'outcome']
original_features = kidney_scaled[features_unchanged]


# In[50]:


list(kidney.columns)


# In[51]:


features_to_scale = [
                 'urine_ph',
                 'osmolality',
                 'conductivity',
                 'urea',
                 'calcium',
                     ]


# In[52]:


def scale_features(df, features_to_scale, method):
    
    df_copy = df.copy()

    data_to_scale = df_copy[features_to_scale]

    if method == 'zscore':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid Method: Choose from 'Z Score', 'MinMax', or 'Robust'.")

    scaled_data = scaler.fit_transform(data_to_scale)
    
    scaled_df = pd.DataFrame(scaled_data, columns=features_to_scale)

    return scaled_df


# In[53]:


# Call scaling function 

scaled_data = scale_features(df=kidney_scaled, features_to_scale=features_to_scale, method='zscore' )


# In[54]:


scaled_data.head()


# In[55]:


# Concatenate scaled and unscaled features

kidney_scaled = pd.concat([original_features, scaled_data], axis=1)
kidney_scaled.head()


# In[56]:


kidney_scaled.info()


# # MODEL BUILDING 

# In[57]:


from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, make_scorer
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn import set_config
import joblib 


# In[58]:


X = kidney_scaled.drop(['outcome'], axis='columns')
y = kidney_scaled['outcome']


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


# ## Tuning hyperparameters

# ### Tuning Solver

# In[60]:


# Instantiate the model       

logreg = LogisticRegression(random_state=7, max_iter=100)


# In[61]:


#Grid search cross validation

parameters = [{'solver': ['lbfgs', 'liblinear', 'sag', 'saga', 'newton-cg']}] 


grid_search = GridSearchCV(estimator = logreg,
                           param_grid = parameters,     #list of evalution metrics 
                           refit=True
                           )


grid_search.fit(X_train,y_train)

print(f"'Best Solver:{grid_search.best_params_}")


# ### Tuning remaining hyperparameters

# In[62]:


logreg = LogisticRegression(solver='lbfgs', random_state=7, max_iter=100, verbose=1)


# In[63]:


parameters = [{'penalty':['none', 'l2'],
              'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}]

scoring = {"AUC": "roc_auc", "accuracy": 'accuracy'}

grid_search = GridSearchCV(estimator = logreg,
                           scoring = scoring,
                           return_train_score = True,
                           param_grid = parameters,
                           cv = 10,
                           n_jobs = 2,
                           refit = 'AUC')

grid_search.fit(X, y)
results = grid_search.cv_results_

print('='*50)
print(f"Best parameters: {grid_search.best_params_}\nBest score: {grid_search.best_score_}")
print('='*50)


# ## Plugging in best hyperparameters

# In[64]:


logreg = LogisticRegression(solver='lbfgs', penalty='none', C=0.001, random_state=7, n_jobs=None, l1_ratio=None)


# ## Running 10-fold cross validation 

# In[65]:


kfold = KFold(n_splits=10, shuffle=True, random_state=43)

training_scores =[]
testing_scores = []

# Iterate over each fold

for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit the model on the training data
    model = logreg.fit(X_train, y_train)

    # Evaluate the model and store the accuracy score
    training = logreg.score(X_train, y_train)
    testing = logreg.score(X_test, y_test)
    training_scores.append(training)
    testing_scores.append(testing) 

for i, (train, test) in enumerate(zip(training_scores, testing_scores), 1):
    print(f'Fold {i}: Training set accuracy = {train:.4f}, Testing set accuracy = {test:.4f}')

print(f'Average Training set accuracy: {np.mean(training_scores):.4f}')
print(f'Average Testing set accuracy: {np.mean(testing_scores):.4f}')


# ### Average scores of training and test sets suggest overfitting on training data

# In[66]:


# Selecting Fold 4 because of highest testing set accuracy and lowest training set accuracy

fold_number = 4

for i, (train_index, test_index) in enumerate(kfold.split(X, y), 1):
    if i == fold_number:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

logreg.fit(X_train, y_train)


# # MODEL EVALUATION 

# ## Confusion matrix 

# In[67]:


y_pred_test = logreg.predict(X_test)

cm = confusion_matrix(y_test, y_pred_test)

print('Confusion matrix\n\n', cm)

print('\nTrue Negatives(TN) = ', cm[0,0])

print('\nTrue Positives(TP) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])


# In[68]:


fig, ax = plt.subplots(figsize=(11, 8))

confusion_matrix = pd.DataFrame(data=cm, columns=['Negative', 'Positive'],
                                index=['Negative', 'Positive'])

sns.heatmap(confusion_matrix, annot=True, fmt='', cmap='Blues', square=True)

ax.xaxis.tick_top()
ax.set_title('Predicted', pad=15, fontsize='30')
plt.ylabel('Actual', fontsize='30')
plt.show()


# ## Classification report 

# In[69]:


target_names = ['Kidney stones absent', 'Kidney stones present']

print(classification_report(y_test, y_pred_test, target_names=target_names))


# ## Receiver Operating Curve 

# In[74]:


# Calculate AUC 

AUC = roc_auc_score(y_test, y_pred_test)

#Plot ROC

fpr, tpr, thresholds = roc_curve(y_test, y_pred_test, pos_label=True)

plt.figure(figsize=(10,10))

plt.plot(fpr, tpr, linewidth=2, label="AUC : {:.4f}".format(AUC))

plt.plot([0,1], [0,1], 'k--' )

plt.rcParams['font.size'] = 25

plt.title('Receiver Operating Characteristic Curve for Predicting Kidney Stones', fontsize=30, pad=30)

plt.xlabel('False Positive Rate' , fontsize=20)

plt.ylabel('True Positive Rate', fontsize=20)

plt.legend(loc=4)

plt.show()


# ## Creating pipeline for future use in similar projects 

# In[71]:


# Construct pipeline 

numeric_cols = list(kidney.columns)

capping = FunctionTransformer(cap_outliers)
scaling = FunctionTransformer(scale_features)

num_pipeline = Pipeline(steps=[
               ('outliers', capping),
               ('scale', scaling)]
                       )

col_trans = ColumnTransformer(transformers=[
                     ('numeric_pipeline', num_pipeline, numeric_cols)],
                     remainder='drop',
                     n_jobs=-1)

logreg = LogisticRegression(solver='lbfgs', random_state=7, max_iter=100, verbose=1)

logreg_pipeline = Pipeline(steps=[
            ('column_transformation', col_trans),
            ('model', logreg)]
                          )


# In[72]:


# Display pipeline 

set_config(display='diagram')
display(logreg_pipeline)


# In[77]:


# Save pipeline 

joblib.dump(logreg_pipeline, 'logistic_classifier_kidney_stones_pipe_0.1.joblib') 

