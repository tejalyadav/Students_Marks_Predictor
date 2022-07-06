from flask import Flask,request, render_template
import numpy as np
import pickle

#initialize the flask app
app = Flask(__name__)

#open the pickle file in the read mode
linear = pickle.load(open('linear.pkl', 'rb'))

@app.route('/')
def home():
    
        return render_template('index.html')
@app.route('/predict',methods=['POST'])
def predict():
        
    
        int_features=[int (x) for x in request.form.values()]
        final_features=np.array(int_features)
        final_features=final_features.reshape(1,-1)
        prediction = linear.predict(final_features)
       
        output = prediction[0]
        print("Marks={}".format(output))
    
        return render_template('index.html', prediction_text='Marks={}'.format(output))
 
if __name__=='__main__':
     app.run(debug=True)
