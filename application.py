import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
model = pickle.load(open('model.pkl', 'rb'))
app = Flask(__name__)
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    pred_prob = model.predict_proba(np.array(final_features))
    
    pred_probalility_score = {"Normal":str(pred_prob[0][1]*100)+'%',"Altered":str(pred_prob[0][0]*100)+'%'}
    if output==0:        
        return render_template('index.html', prediction_text='Altered',pred_probalility_score=pred_probalility_score)
    else:
        return render_template('index.html', prediction_text='Normal',pred_probalility_score=pred_probalility_score)

if __name__ == "__main__":
    app.run()