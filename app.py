from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('WineQualityModel.pkl', 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        FA = float(request.form['fixed acidity'])
        VA=float(request.form['volatile acidity'])
        CA=float(request.form['citric acid'])
        RS = float(request.form["residual sugar"])
        Chlorides=float(request.form['chlorides'])
        FSD=float(request.form['free sulfur dioxide'])
        density = float(request.form['density'])
        pH=float(request.form['pH'])
        sulphates=float(request.form['sulphates'])
        alcohol=float(request.form['alcohol'])
        type_white=int(request.form['type_white'])
            
        data = {
            'fixed acidity': [FA],
            'volatile acidity': [VA],
            'citric acid': [CA],
            'residual sugar':[RS],
            'chlorides': [Chlorides],
            'free sulfur dioxide': [FSD],
            'density': [density],
            'pH': [pH] ,
            'sulphates': [sulphates],
            'alcohol': [alcohol],
            'type_white': [type_white]
        }
        
        data1 = pd.DataFrame(data)
        # print(data1)
        prediction=model.predict(data1)
        if prediction[0]==1:
            txt = "Nice Quality"
        elif prediction[0]==0:
            txt = "Bad Quality"

        return render_template('index.html',prediction = txt)
        
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)