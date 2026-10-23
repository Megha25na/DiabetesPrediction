import pandas as pd
import joblib
from flask import Flask, request, jsonify

#Load the trained models (created by train_model.py)
saved=joblib.load("model.joblib")
columns=saved["columns"]
scaler=saved["scaler"]
knn=saved["knn"]
lr=saved["lr"]
nb=saved["nb"]

app=Flask(__name__,static_folder="static",static_url_path="")

#Open the page
@app.route("/")
def home():
    return app.send_static_file("index.html")

#Receive the form data, predict with all 3 models and send the answers back
@app.route("/predict",methods=["POST"])
def predict():
    d=request.get_json(silent=True) or {}
    try:
        gender=d["gender"]
        smoking=d["smoking_history"]
        age=float(d["age"])
        hypertension=int(d["hypertension"])
        heart_disease=int(d["heart_disease"])
        bmi=float(d["bmi"])
        HbA1c=float(d["HbA1c_level"])
        glucose=float(d["blood_glucose_level"])
    except (KeyError,ValueError,TypeError):
        return jsonify({"error":"Please fill all fields with valid values"}),400

    #Check the values
    if gender not in ["Male","Female"] or smoking not in ["never","former","current","not current","ever","No Info"]:
        return jsonify({"error":"Invalid gender or smoking history"}),400
    if not (1<=age<=120 and 10<=bmi<=100 and 3<=HbA1c<=15 and 50<=glucose<=500):
        return jsonify({"error":"Values are out of range"}),400
    if hypertension not in [0,1] or heart_disease not in [0,1]:
        return jsonify({"error":"Invalid hypertension or heart disease value"}),400

    new_patient=pd.DataFrame([[gender,age,hypertension,heart_disease,smoking,bmi,HbA1c,glucose]],
                             columns=["gender","age","hypertension","heart_disease","smoking_history","bmi","HbA1c_level","blood_glucose_level"])
    #Same encoding and scaling as the training data
    new_patient=pd.get_dummies(new_patient,columns=["gender","smoking_history"]).reindex(columns=columns,fill_value=0).astype(float)
    new_scaled=scaler.transform(new_patient)

    results=[]
    for name,model,row in [("Logistic Regression",lr,new_scaled),("KNN",knn,new_scaled),("Naive Bayes",nb,new_patient)]:
        results.append({"name":name,
                        "prediction":int(model.predict(row)[0]),
                        "probability":round(float(model.predict_proba(row)[0][1])*100,1)})
    return jsonify({"results":results})

if __name__=="__main__":
    print("Open http://127.0.0.1:5000 in your browser")
    app.run()
