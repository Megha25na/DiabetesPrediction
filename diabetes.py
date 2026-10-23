import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score

#Load dataset
df=pd.read_csv("diabetes_prediction_dataset.csv")
print(df.head())
df.info()
print(df.describe())

#Data cleaning
print("Null values:")
print(df.isnull().sum())
print("Duplicate Rows:",df.duplicated().sum())
df=df.drop_duplicates()
df=df[df["gender"]!="Other"]          #only 18 rows
df=df.reset_index(drop=True)
print("Shape after cleaning:",df.shape)

#EDA - class balance
print(df["diabetes"].value_counts())
print(df["diabetes"].value_counts(normalize=True))
sns.countplot(x="diabetes",data=df)
plt.title("Class balance")
plt.show()

#EDA - numeric features vs diabetes
fig,axes=plt.subplots(1,4,figsize=(18,4))
for ax,col in zip(axes,["age","bmi","HbA1c_level","blood_glucose_level"]):
    sns.boxplot(x="diabetes",y=col,data=df,ax=ax)
plt.tight_layout()
plt.show()

#EDA - diabetes rate in each category
for col in ["gender","smoking_history","hypertension","heart_disease"]:
    print(df.groupby(col)["diabetes"].mean().round(3))
    print()

#EDA - correlation
plt.figure(figsize=(8,6))
sns.heatmap(df[["age","hypertension","heart_disease","bmi","HbA1c_level","blood_glucose_level","diabetes"]].corr(),annot=True,fmt=".2f",cmap="coolwarm")
plt.show()

#Important pattern in this dataset
print("HbA1c > 6.6 :",df[df["HbA1c_level"]>6.6]["diabetes"].mean())
print("Glucose > 200 :",df[df["blood_glucose_level"]>200]["diabetes"].mean())
print("HbA1c <= 5.0 :",df[df["HbA1c_level"]<=5.0]["diabetes"].mean())
print("Glucose <= 100 :",df[df["blood_glucose_level"]<=100]["diabetes"].mean())

#Convert text columns (gender, smoking_history) to numbers
data=pd.get_dummies(df,columns=["gender","smoking_history"],drop_first=True)
X=data.drop(columns=["diabetes"]).astype(float)
y=data["diabetes"]

#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

#Scaling (needed for KNN and Logistic Regression)
scaler=StandardScaler()
X_train_s=scaler.fit_transform(X_train)
X_test_s=scaler.transform(X_test)

#Baseline: accuracy if we always predict "No diabetes"
print("Baseline accuracy:",1-y_test.mean())

#KNN
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_s,y_train)
y_pred_knn=knn.predict(X_test_s)
print("\nKNN")
print("Accuracy:",accuracy_score(y_test,y_pred_knn))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred_knn,labels=[0,1]))
print("Precision:",precision_score(y_test,y_pred_knn))
print("Recall:",recall_score(y_test,y_pred_knn))
print("F1 Score:",f1_score(y_test,y_pred_knn))

#Logistic regression
lr=LogisticRegression(max_iter=2000)
lr.fit(X_train_s,y_train)
y_pred_lr=lr.predict(X_test_s)
print("\nLogistic Regression")
print("Accuracy:",accuracy_score(y_test,y_pred_lr))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred_lr,labels=[0,1]))
print("Precision:",precision_score(y_test,y_pred_lr))
print("Recall:",recall_score(y_test,y_pred_lr))
print("F1 Score:",f1_score(y_test,y_pred_lr))

#Naive Bayes (uses Bayes theorem, scaling not needed)
Model_NB=GaussianNB()
Model_NB.fit(X_train,y_train)
y_pred_nb=Model_NB.predict(X_test)
print("\nNaive Bayes")
print("Accuracy:",accuracy_score(y_test,y_pred_nb))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred_nb,labels=[0,1]))
print("Precision:",precision_score(y_test,y_pred_nb))
print("Recall:",recall_score(y_test,y_pred_nb))
print("F1 Score:",f1_score(y_test,y_pred_nb))

#NB Decision Boundary (HbA1c and glucose)
X_data=X_train[["HbA1c_level","blood_glucose_level"]]
Model=GaussianNB()
Model.fit(X_data,y_train)
DecisionBoundaryDisplay.from_estimator(Model,X_data,response_method="predict",alpha=0.3,cmap="Set1")
sample=X_data.sample(2000,random_state=42)
plt.scatter(sample["HbA1c_level"],sample["blood_glucose_level"],c=y_train.loc[sample.index],cmap="Set1",s=8)
plt.xlabel("HbA1c level")
plt.ylabel("Blood glucose level")
plt.title("NB Decision Boundary")
plt.show()

#Prediction for a new patient
age=float(input("Enter the age: "))
hypertension=int(input("Enter hypertension (Yes=1,No=0): "))
heart_disease=int(input("Enter heart disease (Yes=1,No=0): "))
bmi=float(input("Enter the BMI: "))
HbA1c=float(input("Enter the HbA1c level: "))
glucose=int(input("Enter the blood glucose level: "))
gender=input("Enter the gender (Male/Female): ")
smoking=input("Enter smoking history (never/former/current/not current/ever/No Info): ")
new_patient=pd.DataFrame([[gender,age,hypertension,heart_disease,smoking,bmi,HbA1c,glucose]],
                         columns=["gender","age","hypertension","heart_disease","smoking_history","bmi","HbA1c_level","blood_glucose_level"])
#Same encoding as the training data
new_patient=pd.get_dummies(new_patient,columns=["gender","smoking_history"]).reindex(columns=X.columns,fill_value=0).astype(float)
new_scaled=scaler.transform(new_patient)
result={0:"Not Diabetic",1:"Diabetic"}
print("\nKNN:",result[knn.predict(new_scaled)[0]])
print("Logistic Regression:",result[lr.predict(new_scaled)[0]])
print("Naive Bayes:",result[Model_NB.predict(new_patient)[0]])
