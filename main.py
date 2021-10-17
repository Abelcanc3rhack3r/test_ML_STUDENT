import os

import joblib
from flask import Flask, render_template, request
import numpy as np
app= Flask(__name__)

@app.route("/")
def home():
    return render_template("input3.html")

@app.route ("/predict", methods=["POST"])
def predict():
    clf2 = joblib.load("./models/student_pass_predictor.pkl")
    # extract the sex, marks and age from the form
    age = request.form["age"]
    sex = request.form["sex"]
    marks = request.form["marks"]
    ple = joblib.load("./models/ple.pkl")
    sext = joblib.load("./models/sex.pkl")
    # convert sex to a number using sex transform

    sex_transformed = sext.transform([sex[0]])
    # convert the age to a number using age transform

    values = [ sex_transformed, age, marks]
    print("values:", values)
    # convert the values into a numpy array
    values = np.array(values).reshape((1, -1))
    # get the prediction
    prediction = clf2.predict(values)
    # convert the integer prediction into string using ple inverse
    print("passed", prediction[0])
    prediction = ple.inverse_transform(prediction)
    print("passed", prediction[0])
    # return the prediction
    result= "pass" if prediction[0]=="yes" else "fail"
    return render_template("results_page.html", prediction=result, inputt= [age, sex, marks])

# create a submit_image route
@app.route('/submit_image', methods=['POST'])
def submit_image():
    file = request.files['file']
    f = os.path.join(app.root_path, 'static', 'uploads', file.filename)
    file.save(f)
    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    #file.save(f)
    #cv= cv2.imread(f)
    #cv2.imshow("image", cv)
    #cv2.waitKey(0)
    # show image on screen

    return render_template('show.html', url="../static/uploads/1602.png")

@app.route("/submit", methods =["POST"])
def submit():
    if request.method=="POST":
        name= request.form["username"]
    return render_template("submit.html", name= name)
# run app
if __name__ == '__main__':
    app.run(debug=True)
