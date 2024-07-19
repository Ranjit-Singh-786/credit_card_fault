import numpy as np
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
# prediction function 
def ValuePredictor(to_predict_list): 
    # return 1
    to_predict = np.array(to_predict_list).reshape(1, 7) 
    loaded_model = joblib.load('models\XGBClassifier.lb') 
    result = loaded_model.predict(to_predict) 
    return result[0]     
    
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)
    if int(result)== 1:
        prediction ='Given transaction is fradulent'
    else:
        prediction ='Given transaction is NOT fradulent'            
    return render_template("result.html", prediction = prediction) 
    
    
    
    
    
                  
if __name__ == "__main__":
    app.run(debug=True)
