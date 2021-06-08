# Credit Risk Resampling Techniques


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from sklearn.preprocessing import LabelEncoder
```

# Read the CSV into DataFrame


```python
# Load the data
file_path = Path('Resources/lending_data.csv')
df = pd.read_csv(file_path)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_size</th>
      <th>interest_rate</th>
      <th>homeowner</th>
      <th>borrower_income</th>
      <th>debt_to_income</th>
      <th>num_of_accounts</th>
      <th>derogatory_marks</th>
      <th>total_debt</th>
      <th>loan_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10700.0</td>
      <td>7.672</td>
      <td>own</td>
      <td>52800</td>
      <td>0.431818</td>
      <td>5</td>
      <td>1</td>
      <td>22800</td>
      <td>low_risk</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8400.0</td>
      <td>6.692</td>
      <td>own</td>
      <td>43600</td>
      <td>0.311927</td>
      <td>3</td>
      <td>0</td>
      <td>13600</td>
      <td>low_risk</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9000.0</td>
      <td>6.963</td>
      <td>rent</td>
      <td>46100</td>
      <td>0.349241</td>
      <td>3</td>
      <td>0</td>
      <td>16100</td>
      <td>low_risk</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10700.0</td>
      <td>7.664</td>
      <td>own</td>
      <td>52700</td>
      <td>0.430740</td>
      <td>5</td>
      <td>1</td>
      <td>22700</td>
      <td>low_risk</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10800.0</td>
      <td>7.698</td>
      <td>mortgage</td>
      <td>53000</td>
      <td>0.433962</td>
      <td>5</td>
      <td>1</td>
      <td>23000</td>
      <td>low_risk</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Fitting and encoding the columns with the LabelEncoder
le = LabelEncoder()

# Encoding homeowner column
le.fit(df["homeowner"])
df["homeowner"] = le.transform(df["homeowner"])

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_size</th>
      <th>interest_rate</th>
      <th>homeowner</th>
      <th>borrower_income</th>
      <th>debt_to_income</th>
      <th>num_of_accounts</th>
      <th>derogatory_marks</th>
      <th>total_debt</th>
      <th>loan_status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10700.0</td>
      <td>7.672</td>
      <td>1</td>
      <td>52800</td>
      <td>0.431818</td>
      <td>5</td>
      <td>1</td>
      <td>22800</td>
      <td>low_risk</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8400.0</td>
      <td>6.692</td>
      <td>1</td>
      <td>43600</td>
      <td>0.311927</td>
      <td>3</td>
      <td>0</td>
      <td>13600</td>
      <td>low_risk</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9000.0</td>
      <td>6.963</td>
      <td>2</td>
      <td>46100</td>
      <td>0.349241</td>
      <td>3</td>
      <td>0</td>
      <td>16100</td>
      <td>low_risk</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10700.0</td>
      <td>7.664</td>
      <td>1</td>
      <td>52700</td>
      <td>0.430740</td>
      <td>5</td>
      <td>1</td>
      <td>22700</td>
      <td>low_risk</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10800.0</td>
      <td>7.698</td>
      <td>0</td>
      <td>53000</td>
      <td>0.433962</td>
      <td>5</td>
      <td>1</td>
      <td>23000</td>
      <td>low_risk</td>
    </tr>
  </tbody>
</table>
</div>



# Split the Data into Training and Testing


```python
# Create our features
X= df.drop(columns="loan_status")

# Create our target
y=df["loan_status"]
X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_size</th>
      <th>interest_rate</th>
      <th>homeowner</th>
      <th>borrower_income</th>
      <th>debt_to_income</th>
      <th>num_of_accounts</th>
      <th>derogatory_marks</th>
      <th>total_debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10700.0</td>
      <td>7.672</td>
      <td>1</td>
      <td>52800</td>
      <td>0.431818</td>
      <td>5</td>
      <td>1</td>
      <td>22800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8400.0</td>
      <td>6.692</td>
      <td>1</td>
      <td>43600</td>
      <td>0.311927</td>
      <td>3</td>
      <td>0</td>
      <td>13600</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9000.0</td>
      <td>6.963</td>
      <td>2</td>
      <td>46100</td>
      <td>0.349241</td>
      <td>3</td>
      <td>0</td>
      <td>16100</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10700.0</td>
      <td>7.664</td>
      <td>1</td>
      <td>52700</td>
      <td>0.430740</td>
      <td>5</td>
      <td>1</td>
      <td>22700</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10800.0</td>
      <td>7.698</td>
      <td>0</td>
      <td>53000</td>
      <td>0.433962</td>
      <td>5</td>
      <td>1</td>
      <td>23000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>77531</th>
      <td>19100.0</td>
      <td>11.261</td>
      <td>1</td>
      <td>86600</td>
      <td>0.653580</td>
      <td>12</td>
      <td>2</td>
      <td>56600</td>
    </tr>
    <tr>
      <th>77532</th>
      <td>17700.0</td>
      <td>10.662</td>
      <td>0</td>
      <td>80900</td>
      <td>0.629172</td>
      <td>11</td>
      <td>2</td>
      <td>50900</td>
    </tr>
    <tr>
      <th>77533</th>
      <td>17600.0</td>
      <td>10.595</td>
      <td>2</td>
      <td>80300</td>
      <td>0.626401</td>
      <td>11</td>
      <td>2</td>
      <td>50300</td>
    </tr>
    <tr>
      <th>77534</th>
      <td>16300.0</td>
      <td>10.068</td>
      <td>0</td>
      <td>75300</td>
      <td>0.601594</td>
      <td>10</td>
      <td>2</td>
      <td>45300</td>
    </tr>
    <tr>
      <th>77535</th>
      <td>15600.0</td>
      <td>9.742</td>
      <td>0</td>
      <td>72300</td>
      <td>0.585062</td>
      <td>9</td>
      <td>2</td>
      <td>42300</td>
    </tr>
  </tbody>
</table>
<p>77536 rows × 8 columns</p>
</div>




```python
X.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_size</th>
      <th>interest_rate</th>
      <th>homeowner</th>
      <th>borrower_income</th>
      <th>debt_to_income</th>
      <th>num_of_accounts</th>
      <th>derogatory_marks</th>
      <th>total_debt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>77536.000000</td>
      <td>77536.000000</td>
      <td>77536.000000</td>
      <td>77536.000000</td>
      <td>77536.000000</td>
      <td>77536.000000</td>
      <td>77536.000000</td>
      <td>77536.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>9805.562577</td>
      <td>7.292333</td>
      <td>0.606144</td>
      <td>49221.949804</td>
      <td>0.377318</td>
      <td>3.826610</td>
      <td>0.392308</td>
      <td>19221.949804</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2093.223153</td>
      <td>0.889495</td>
      <td>0.667811</td>
      <td>8371.635077</td>
      <td>0.081519</td>
      <td>1.904426</td>
      <td>0.582086</td>
      <td>8371.635077</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5000.000000</td>
      <td>5.250000</td>
      <td>0.000000</td>
      <td>30000.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8700.000000</td>
      <td>6.825000</td>
      <td>0.000000</td>
      <td>44800.000000</td>
      <td>0.330357</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>14800.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9500.000000</td>
      <td>7.172000</td>
      <td>1.000000</td>
      <td>48100.000000</td>
      <td>0.376299</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>18100.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>10400.000000</td>
      <td>7.528000</td>
      <td>1.000000</td>
      <td>51400.000000</td>
      <td>0.416342</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>21400.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>23800.000000</td>
      <td>13.235000</td>
      <td>2.000000</td>
      <td>105200.000000</td>
      <td>0.714829</td>
      <td>16.000000</td>
      <td>3.000000</td>
      <td>75200.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check the balance of our target values
y.value_counts()
```




    low_risk     75036
    high_risk     2500
    Name: loan_status, dtype: int64




```python
# Create X_train, X_test, y_train, y_test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X,
                                                   y,
                                                   random_state=1,
                                                   stratify=y)
X_train.shape
```




    (58152, 8)



## Data Pre-Processing

Scale the training and testing data using the `StandardScaler` from `sklearn`. Remember that when scaling the data, you only scale the features data (`X_train` and `X_testing`).


```python
# Create the StandardScaler instance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
```


```python
# Fit the Standard Scaler with the training data
# When fitting scaling functions, only train on the training dataset
X_scaler = scaler.fit(X_train)
```


```python
# Scale the training and testing data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```

# Simple Logistic Regression


```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs', random_state=1)
model.fit(X_train, y_train)
```




    LogisticRegression(random_state=1)




```python
# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
y_pred = model.predict(X_test_scaled)
bas=balanced_accuracy_score(y_test, y_pred)
print(bas)
```

    0.8041461911615757
    


```python
# Display the confusion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
```




    array([[  622,     3],
           [ 7258, 11501]], dtype=int64)




```python
# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.08      1.00      0.61      0.15      0.78      0.63       625
       low_risk       1.00      0.61      1.00      0.76      0.78      0.59     18759
    
    avg / total       0.97      0.63      0.98      0.74      0.78      0.59     19384
    
    

# Oversampling

In this section, you will compare two oversampling algorithms to determine which algorithm results in the best performance. You will oversample the data using the naive random oversampling algorithm and the SMOTE algorithm. For each algorithm, be sure to complete the folliowing steps:

1. View the count of the target classes using `Counter` from the collections library. 
3. Use the resampled data to train a logistic regression model.
3. Calculate the balanced accuracy score from sklearn.metrics.
4. Print the confusion matrix from sklearn.metrics.
5. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.

Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests

### Naive Random Oversampling


```python
# Resample the training data with the RandomOversampler
from imblearn.over_sampling import RandomOverSampler
# View the count of target classes with Counter
ros = RandomOverSampler(random_state=1)
X_resampled_ros, y_resampled_ros = ros.fit_resample(X_train_scaled, y_train)
Counter(y_resampled_ros)
```




    Counter({'low_risk': 56277, 'high_risk': 56277})




```python
# Train the Logistic Regression model using the resampled data
from sklearn.linear_model import LogisticRegression

model_ros = LogisticRegression(solver='lbfgs', random_state=1)
model_ros.fit(X_resampled_ros, y_resampled_ros)
```




    LogisticRegression(random_state=1)




```python
# Calculated the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score

y_pred_ros = model_ros.predict(X_test_scaled)

bas_ros=balanced_accuracy_score(y_test, y_pred_ros)
print(bas_ros)
```

    0.9946414201183431
    


```python
# Display the confusion matrix
from sklearn.metrics import confusion_matrix

cm_ros = confusion_matrix(y_test, y_pred_ros)
cm_df_ros = pd.DataFrame(
    cm_ros, index=["Actual High Risk", "Actual Low Risk"], columns=["Predicted High Risk", "Predicted Low Risk"]
)
cm_df_ros
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted High Risk</th>
      <th>Predicted Low Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual High Risk</th>
      <td>622</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Actual Low Risk</th>
      <td>111</td>
      <td>18648</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced

print(classification_report_imbalanced(y_test, y_pred_ros))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.85      1.00      0.99      0.92      0.99      0.99       625
       low_risk       1.00      0.99      1.00      1.00      0.99      0.99     18759
    
    avg / total       0.99      0.99      1.00      0.99      0.99      0.99     19384
    
    

### SMOTE Oversampling


```python
# Resample the training data with SMOTE
from imblearn.over_sampling import SMOTE

X_resampled_smote, y_resampled_smote = SMOTE(random_state=1, sampling_strategy=1.0).fit_resample(
    X_train_scaled, y_train
)
from collections import Counter
# View the count of target classes with Counter
Counter(y_resampled_smote)
```




    Counter({'low_risk': 56277, 'high_risk': 56277})




```python
# Train the Logistic Regression model using the resampled data
model_smote = LogisticRegression(solver='lbfgs', random_state=1)
model_smote.fit(X_resampled_smote, y_resampled_smote)
```




    LogisticRegression(random_state=1)




```python
# Calculated the balanced accuracy score
y_pred_smote = model_smote.predict(X_test_scaled)
bas_smote=balanced_accuracy_score(y_test, y_pred_smote)
print(bas_smote)
```

    0.9946414201183431
    


```python
# Display the confusion matrix
cm_smote = confusion_matrix(y_test, y_pred_smote)
cm_df_smote = pd.DataFrame(
    cm_smote, index=["Actual High Risk", "Actual Low Risk"], columns=["Predicted High Risk", "Predicted Low Risk"]
)
cm_df_smote
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted High Risk</th>
      <th>Predicted Low Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual High Risk</th>
      <td>622</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Actual Low Risk</th>
      <td>111</td>
      <td>18648</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, y_pred_smote))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.85      1.00      0.99      0.92      0.99      0.99       625
       low_risk       1.00      0.99      1.00      1.00      0.99      0.99     18759
    
    avg / total       0.99      0.99      1.00      0.99      0.99      0.99     19384
    
    

# Undersampling

In this section, you will test an undersampling algorithm to determine which algorithm results in the best performance compared to the oversampling algorithms above. You will undersample the data using the Cluster Centroids algorithm and complete the folliowing steps:

1. View the count of the target classes using `Counter` from the collections library. 
3. Use the resampled data to train a logistic regression model.
3. Calculate the balanced accuracy score from sklearn.metrics.
4. Display the confusion matrix from sklearn.metrics.
5. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.

Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests


```python
# Resample the data using the ClusterCentroids resampler
from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(random_state=1)
X_resampled_cc, y_resampled_cc = cc.fit_resample(X_train_scaled, y_train)
# View the count of target classes with Counter
Counter(y_resampled_cc)
```




    Counter({'high_risk': 1875, 'low_risk': 1875})




```python
# Train the Logistic Regression model using the resampled data
from sklearn.linear_model import LogisticRegression
model_cc = LogisticRegression(solver='lbfgs', random_state=1)
model_cc.fit(X_resampled_cc, y_resampled_cc)
```




    LogisticRegression(random_state=1)




```python
# Calculate the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
y_pred_cc = model_cc.predict(X_test_scaled)
bas_cc=balanced_accuracy_score(y_test, y_pred_cc)
print(bas_cc)
```

    0.9932813049736127
    


```python
# Display the confusion matrix
from sklearn.metrics import confusion_matrix
cm_cc = confusion_matrix(y_test, y_pred_cc)
cm_df_cc = pd.DataFrame(
    cm_cc, index=["Actual High Risk", "Actual Low Risk"], columns=["Predicted High Risk", "Predicted Low Risk"]
)
cm_df_cc
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted High Risk</th>
      <th>Predicted Low Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual High Risk</th>
      <td>620</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Actual Low Risk</th>
      <td>102</td>
      <td>18657</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred_cc))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.86      0.99      0.99      0.92      0.99      0.99       625
       low_risk       1.00      0.99      0.99      1.00      0.99      0.99     18759
    
    avg / total       1.00      0.99      0.99      0.99      0.99      0.99     19384
    
    

# Combination (Over and Under) Sampling

In this section, you will test a combination over- and under-sampling algorithm to determine if the algorithm results in the best performance compared to the other sampling algorithms above. You will resample the data using the SMOTEENN algorithm and complete the folliowing steps:

1. View the count of the target classes using `Counter` from the collections library. 
3. Use the resampled data to train a logistic regression model.
3. Calculate the balanced accuracy score from sklearn.metrics.
4. Display the confusion matrix from sklearn.metrics.
5. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.

Note: Use a random state of 1 for each sampling algorithm to ensure consistency between tests


```python
# Resample the training data with SMOTEENN
from imblearn.combine import SMOTEENN

smote_enn = SMOTEENN(random_state=0)
X_resampled_se, y_resampled_se= smote_enn.fit_resample(X_train_scaled, y_train)

# View the count of target classes with Counter
Counter(y_resampled_se)
```




    Counter({'high_risk': 55636, 'low_risk': 55913})




```python
# Train the Logistic Regression model using the resampled data
from sklearn.linear_model import LogisticRegression
model_se = LogisticRegression(solver='lbfgs', random_state=1)
model_se.fit(X_resampled_se, y_resampled_se)
```




    LogisticRegression(random_state=1)




```python
# Calculate the balanced accuracy score
from sklearn.metrics import balanced_accuracy_score
y_pred_se = model_se.predict(X_test_scaled)
bas_se=balanced_accuracy_score(y_test, y_pred_se)
print(bas_se)
```

    0.9946414201183431
    


```python
# Display the confusion matrix
cm_se = confusion_matrix(y_test, y_pred_se)
cm_df_se = pd.DataFrame(
    cm_se, index=["Actual High Risk", "Actual Low Risk"], columns=["Predicted High Risk", "Predicted Low Risk"]
)
cm_df_se
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted High Risk</th>
      <th>Predicted Low Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual High Risk</th>
      <td>622</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Actual Low Risk</th>
      <td>111</td>
      <td>18648</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the imbalanced classification report
from imblearn.metrics import classification_report_imbalanced
print(classification_report_imbalanced(y_test, y_pred_se))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.85      1.00      0.99      0.92      0.99      0.99       625
       low_risk       1.00      0.99      1.00      1.00      0.99      0.99     18759
    
    avg / total       0.99      0.99      1.00      0.99      0.99      0.99     19384
    
    

# Final Questions

1. Which model had the best balanced accuracy score?

   Naive Oversampling, SMOTE and SMOTEENN all had the same balanced accuracy score

2. Which model had the best recall score?

    The Naive Oversampling, SMOTE and SMOTEENN all had the same Recall Scores.
    High Risk: 1.00, Low Risk: 0.99, Total: 0.99

3. Which model had the best geometric mean score?

    All models have the same f1 score.
    High Risk: 0.92, Low Risk: 1.00, Total: 0.99



```python
print(f"Original data set's balanced accuracy score = {bas}")
print(f"-------------------------------------------------------------------")
print(f"Naive overampling balanced accuracy score = {bas_ros} ")
print(f"SMOTE balanced accuracy score = {bas_smote} ")
print(f"SMOTEEN balanced accuracy score = {bas_se} ")

```

    Original data set's balanced accuracy score = 0.8041461911615757
    -------------------------------------------------------------------
    Naive overampling balanced accuracy score = 0.9946414201183431 
    SMOTE balanced accuracy score = 0.9946414201183431 
    SMOTEEN balanced accuracy score = 0.9946414201183431 
    


# Ensemble Learning

## Initial Imports


```python
import warnings
warnings.filterwarnings('ignore')
```


```python
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from imblearn.metrics import classification_report_imbalanced
```

## Read the CSV and Perform Basic Data Cleaning


```python
# Load the data
file_path = Path('Resources/LoanStats_2019Q1.csv')
df = pd.read_csv(file_path)
df = pd.get_dummies(df, columns=["home_ownership", "verification_status","hardship_flag","debt_settlement_flag","issue_d","pymnt_plan","application_type","initial_list_status","next_pymnt_d"])

# Preview the data
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>loan_status</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>inq_last_6mths</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>...</th>
      <th>issue_d_Feb-2019</th>
      <th>issue_d_Jan-2019</th>
      <th>issue_d_Mar-2019</th>
      <th>pymnt_plan_n</th>
      <th>application_type_Individual</th>
      <th>application_type_Joint App</th>
      <th>initial_list_status_f</th>
      <th>initial_list_status_w</th>
      <th>next_pymnt_d_Apr-2019</th>
      <th>next_pymnt_d_May-2019</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10500.0</td>
      <td>0.1719</td>
      <td>375.35</td>
      <td>66000.0</td>
      <td>low_risk</td>
      <td>27.24</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>25000.0</td>
      <td>0.2000</td>
      <td>929.09</td>
      <td>105000.0</td>
      <td>low_risk</td>
      <td>20.23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>17.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20000.0</td>
      <td>0.2000</td>
      <td>529.88</td>
      <td>56000.0</td>
      <td>low_risk</td>
      <td>24.26</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10000.0</td>
      <td>0.1640</td>
      <td>353.55</td>
      <td>92000.0</td>
      <td>low_risk</td>
      <td>31.44</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22000.0</td>
      <td>0.1474</td>
      <td>520.39</td>
      <td>52000.0</td>
      <td>low_risk</td>
      <td>18.76</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 96 columns</p>
</div>



## Split the Data into Training and Testing


```python
# Create our features
X = df.drop(columns="loan_status")

# Create our target
y =df["loan_status"]
y.head()
```




    0    low_risk
    1    low_risk
    2    low_risk
    3    low_risk
    4    low_risk
    Name: loan_status, dtype: object




```python
X.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>loan_amnt</th>
      <th>int_rate</th>
      <th>installment</th>
      <th>annual_inc</th>
      <th>dti</th>
      <th>delinq_2yrs</th>
      <th>inq_last_6mths</th>
      <th>open_acc</th>
      <th>pub_rec</th>
      <th>revol_bal</th>
      <th>...</th>
      <th>issue_d_Feb-2019</th>
      <th>issue_d_Jan-2019</th>
      <th>issue_d_Mar-2019</th>
      <th>pymnt_plan_n</th>
      <th>application_type_Individual</th>
      <th>application_type_Joint App</th>
      <th>initial_list_status_f</th>
      <th>initial_list_status_w</th>
      <th>next_pymnt_d_Apr-2019</th>
      <th>next_pymnt_d_May-2019</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>6.881700e+04</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>...</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.0</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
      <td>68817.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>16677.594562</td>
      <td>0.127718</td>
      <td>480.652863</td>
      <td>8.821371e+04</td>
      <td>21.778153</td>
      <td>0.217766</td>
      <td>0.497697</td>
      <td>12.587340</td>
      <td>0.126030</td>
      <td>17604.142828</td>
      <td>...</td>
      <td>0.371696</td>
      <td>0.451066</td>
      <td>0.177238</td>
      <td>1.0</td>
      <td>0.860340</td>
      <td>0.139660</td>
      <td>0.123879</td>
      <td>0.876121</td>
      <td>0.383161</td>
      <td>0.616839</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10277.348590</td>
      <td>0.048130</td>
      <td>288.062432</td>
      <td>1.155800e+05</td>
      <td>20.199244</td>
      <td>0.718367</td>
      <td>0.758122</td>
      <td>6.022869</td>
      <td>0.336797</td>
      <td>21835.880400</td>
      <td>...</td>
      <td>0.483261</td>
      <td>0.497603</td>
      <td>0.381873</td>
      <td>0.0</td>
      <td>0.346637</td>
      <td>0.346637</td>
      <td>0.329446</td>
      <td>0.329446</td>
      <td>0.486161</td>
      <td>0.486161</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1000.000000</td>
      <td>0.060000</td>
      <td>30.890000</td>
      <td>4.000000e+01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>9000.000000</td>
      <td>0.088100</td>
      <td>265.730000</td>
      <td>5.000000e+04</td>
      <td>13.890000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>0.000000</td>
      <td>6293.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>15000.000000</td>
      <td>0.118000</td>
      <td>404.560000</td>
      <td>7.300000e+04</td>
      <td>19.760000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>12068.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>24000.000000</td>
      <td>0.155700</td>
      <td>648.100000</td>
      <td>1.040000e+05</td>
      <td>26.660000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>16.000000</td>
      <td>0.000000</td>
      <td>21735.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>40000.000000</td>
      <td>0.308400</td>
      <td>1676.230000</td>
      <td>8.797500e+06</td>
      <td>999.000000</td>
      <td>18.000000</td>
      <td>5.000000</td>
      <td>72.000000</td>
      <td>4.000000</td>
      <td>587191.000000</td>
      <td>...</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 95 columns</p>
</div>




```python
# Check the balance of our target values
y.value_counts()
```




    low_risk     68470
    high_risk      347
    Name: loan_status, dtype: int64




```python
# Split the X and y into X_train, X_test, y_train, y_test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X,
                                                   y,
                                                   random_state=1,
                                                   stratify=y)
X_train.shape
```




    (51612, 95)



## Data Pre-Processing

Scale the training and testing data using the `StandardScaler` from `sklearn`. Remember that when scaling the data, you only scale the features data (`X_train` and `X_testing`).


```python
# Create the StandardScaler instance
scaler = StandardScaler()
```


```python
# Fit the Standard Scaler with the training data
# When fitting scaling functions, only train on the training dataset
X_scaler = scaler.fit(X_train)
```


```python
# Scale the training and testing data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```

## Ensemble Learners

In this section, you will compare two ensemble algorithms to determine which algorithm results in the best performance. You will train a Balanced Random Forest Classifier and an Easy Ensemble classifier . For each algorithm, be sure to complete the folliowing steps:

1. Train the model using the training data. 
2. Calculate the balanced accuracy score from sklearn.metrics.
3. Display the confusion matrix from sklearn.metrics.
4. Generate a classication report using the `imbalanced_classification_report` from imbalanced-learn.
5. For the Balanced Random Forest Classifier only, print the feature importance sorted in descending order (most important feature to least important) along with the feature score

Note: Use a random state of 1 for each algorithm to ensure consistency between tests

### Balanced Random Forest Classifier


```python
# Create a random forest classifier
rf_model = BalancedRandomForestClassifier(n_estimators=100, random_state=78)
# Resample the training data with the BalancedRandomForestClassifier
rf_model.fit(X_train, y_train)
predictions = rf_model.predict(X_test)
```


```python
# Calculated the balanced accuracy score
acc_score = accuracy_score(y_test, predictions)
print(acc_score)
```

    0.9019471083987213
    


```python
# Display the confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

matrix = confusion_matrix(y_test, predictions)
cm_df = pd.DataFrame(
    matrix, index=["Actual High-Risk", "Actual Low-Risk"], columns=["Predicted High_Risk", "Predicted Low_Risk"])
cm_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Predicted High_Risk</th>
      <th>Predicted Low_Risk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Actual High-Risk</th>
      <td>59</td>
      <td>28</td>
    </tr>
    <tr>
      <th>Actual Low-Risk</th>
      <td>1659</td>
      <td>15459</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, predictions))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.03      0.68      0.90      0.07      0.78      0.60        87
       low_risk       1.00      0.90      0.68      0.95      0.78      0.63     17118
    
    avg / total       0.99      0.90      0.68      0.94      0.78      0.63     17205
    
    


```python
# List the features sorted in descending order by feature importance
sorted(zip(rf_model.feature_importances_, X.columns), reverse=True)
```




    [(0.06949794660629233, 'last_pymnt_amnt'),
     (0.06838768711754954, 'total_rec_prncp'),
     (0.06187727039749519, 'total_rec_int'),
     (0.06140972349105115, 'total_pymnt_inv'),
     (0.050525395963678756, 'total_pymnt'),
     (0.027666832727408705, 'int_rate'),
     (0.018739369529793704, 'issue_d_Jan-2019'),
     (0.017598517096615973, 'max_bal_bc'),
     (0.01747812829661088, 'installment'),
     (0.017316206673222492, 'mths_since_recent_inq'),
     (0.016880829529775085, 'annual_inc'),
     (0.01685849647540544, 'dti'),
     (0.01680383866083915, 'il_util'),
     (0.016283192036219207, 'out_prncp'),
     (0.016018759178375898, 'mths_since_rcnt_il'),
     (0.01591511414338516, 'tot_hi_cred_lim'),
     (0.01481863364673417, 'revol_bal'),
     (0.014567026518848319, 'mo_sin_old_rev_tl_op'),
     (0.014339342937031707, 'bc_util'),
     (0.014210161457150244, 'total_rev_hi_lim'),
     (0.01399891735899933, 'total_bc_limit'),
     (0.013875519140365633, 'out_prncp_inv'),
     (0.013855815551959121, 'mo_sin_old_il_acct'),
     (0.013450759539593782, 'avg_cur_bal'),
     (0.013425112126302565, 'all_util'),
     (0.013207128954509659, 'total_bal_ex_mort'),
     (0.012560984788415797, 'mo_sin_rcnt_rev_tl_op'),
     (0.0120314293643251, 'mths_since_recent_bc'),
     (0.011485050160934733, 'issue_d_Mar-2019'),
     (0.011354713714474975, 'bc_open_to_buy'),
     (0.011332476986650485, 'tot_cur_bal'),
     (0.011316492825868003, 'loan_amnt'),
     (0.010681085608205609, 'num_rev_accts'),
     (0.010359722889775996, 'total_bal_il'),
     (0.0100605686610884, 'total_acc'),
     (0.010027265019606149, 'open_acc'),
     (0.009997825156358095, 'total_il_high_credit_limit'),
     (0.00988440518847866, 'num_il_tl'),
     (0.009358911671045678, 'num_bc_sats'),
     (0.00909940840514533, 'num_bc_tl'),
     (0.008553667061669084, 'num_actv_bc_tl'),
     (0.00847570409102426, 'inq_fi'),
     (0.008413732796439305, 'num_sats'),
     (0.008388938497428063, 'num_actv_rev_tl'),
     (0.008136048421934468, 'num_op_rev_tl'),
     (0.008051425226134007, 'num_rev_tl_bal_gt_0'),
     (0.007893497645678286, 'mo_sin_rcnt_tl'),
     (0.007740999424851808, 'acc_open_past_24mths'),
     (0.0075969971019928, 'inq_last_12m'),
     (0.007445113023707023, 'total_cu_tl'),
     (0.007309979259896024, 'open_act_il'),
     (0.007232151190669385, 'open_rv_24m'),
     (0.007157301975959763, 'open_il_24m'),
     (0.00709804733944831, 'pct_tl_nvr_dlq'),
     (0.006693995963384435, 'issue_d_Feb-2019'),
     (0.006507635998493077, 'num_tl_op_past_12m'),
     (0.006179268922306, 'inq_last_6mths'),
     (0.005711241404505625, 'percent_bc_gt_75'),
     (0.005509503197686365, 'total_rec_late_fee'),
     (0.00513387550110993, 'mort_acc'),
     (0.005123417899530327, 'delinq_2yrs'),
     (0.005101056380067826, 'open_rv_12m'),
     (0.004648108460350013, 'open_il_12m'),
     (0.004618100235955401, 'next_pymnt_d_May-2019'),
     (0.004298876110687663, 'next_pymnt_d_Apr-2019'),
     (0.00423633502641356, 'open_acc_6m'),
     (0.0037508598101456503, 'verification_status_Not Verified'),
     (0.0032387154612916434, 'tot_coll_amt'),
     (0.0026128650367577898, 'home_ownership_OWN'),
     (0.002142238193194172, 'verification_status_Source Verified'),
     (0.002138607240243092, 'verification_status_Verified'),
     (0.0020695020734779208, 'num_accts_ever_120_pd'),
     (0.0016545331782660566, 'home_ownership_MORTGAGE'),
     (0.0015807849005200678, 'application_type_Individual'),
     (0.0015737440150458666, 'home_ownership_RENT'),
     (0.001448392449895477, 'pub_rec'),
     (0.0013313239951401375, 'initial_list_status_f'),
     (0.0010696736846435265, 'num_tl_90g_dpd_24m'),
     (0.0009431347885561211, 'pub_rec_bankruptcies'),
     (0.0009349877264532822, 'application_type_Joint App'),
     (0.0008479294193358862, 'initial_list_status_w'),
     (0.0006337153666595339, 'collections_12_mths_ex_med'),
     (0.00014430761944230657, 'chargeoff_within_12_mths'),
     (7.360128802254023e-05, 'home_ownership_ANY'),
     (0.0, 'tax_liens'),
     (0.0, 'recoveries'),
     (0.0, 'pymnt_plan_n'),
     (0.0, 'policy_code'),
     (0.0, 'num_tl_30dpd'),
     (0.0, 'num_tl_120dpd_2m'),
     (0.0, 'hardship_flag_N'),
     (0.0, 'delinq_amnt'),
     (0.0, 'debt_settlement_flag_N'),
     (0.0, 'collection_recovery_fee'),
     (0.0, 'acc_now_delinq')]



### Easy Ensemble Classifier


```python
# Train the Classifier
eec = EasyEnsembleClassifier(n_estimators=100, random_state=61)

eec.fit(X_train, y_train)
```




    EasyEnsembleClassifier(n_estimators=100, random_state=61)




```python
# Calculated the balanced accuracy score
e_acc_score = balanced_accuracy_score(y_test, predictions)
print(f"Balanced Accuracy Score : {e_acc_score}")
```

    Balanced Accuracy Score : 0.7906226960126665
    


```python
# Display the confusion matrix
cm = confusion_matrix(y_test, predictions)
cm_df = pd.DataFrame(
    cm, index=["Actual 0", "Actual 1"],
    columns=["Predicted 0", "Predicted 1"]
)
display(cm_df)
```


```python
# Print the imbalanced classification report
print(classification_report_imbalanced(y_test, predictions))
```

                       pre       rec       spe        f1       geo       iba       sup
    
      high_risk       0.03      0.68      0.90      0.07      0.78      0.60        87
       low_risk       1.00      0.90      0.68      0.95      0.78      0.63     17118
    
    avg / total       0.99      0.90      0.68      0.94      0.78      0.63     17205
    
    

### Final Questions

1. Which model had the best balanced accuracy score?


```python
print(f"Balanced Random Forest = {acc_score} ")
print(f"Easy Ensembly = {e_acc_score}")

```

    Balanced Random Forest = 0.9019471083987213 
    Easy Ensembly = 0.7906226960126665
    


2. Which model had the best recall score?


```python
print(f"Balanced Random Forest = 0.90 ")
print(f"Easy Ensembly = 0.95")
```

    Balanced Random Forest = 0.90 
    Easy Ensembly = 0.95
    

3. Which model had the best geometric mean score?


```python
print(f"Balanced Random Forest = 0.78 ")
print(f"Easy Ensembly = 0.78")
```

    Balanced Random Forest = 0.78 
    Easy Ensembly = 0.78
    

4. What are the top three features?


```python
sorted(zip(rf_model.feature_importances_, X.columns), reverse=True)[:3]

```




    [(0.06949794660629233, 'last_pymnt_amnt'),
     (0.06838768711754954, 'total_rec_prncp'),
     (0.06187727039749519, 'total_rec_int')]


