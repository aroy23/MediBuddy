from flask import Flask, render_template, request, redirect, url_for, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import csv
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import google.generativeai as genai
from sqlalchemy.exc import IntegrityError
import random
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from io import BytesIO
from reportlab.lib.utils import simpleSplit


def create_csv_file(file_name):
    with open(file_name, 'w', newline='') as csvfile:
        fieldnames = ['patient', 'latest_report']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


genText = None
currentUser = None

def format_markdown(input_string):
    return input_string.replace("*", "")

def printToGemini(arg:str):
    genai.configure(api_key="AIzaSyDVJS1744dqs1WtNEflRiUSEgqGQzhIrRQ")

    # Set up the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
    }

    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

    convo = model.start_chat(history=[])

    convo.send_message(arg)
    global genText 
    genText = convo.last.text

def getDiabetesData():
    data = pd.read_csv('datadb.csv')
    diabetes = pd.read_csv('diabetes.csv')

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data.loc[:, data.columns != 'Outcome'], data['Outcome'], stratify=data['Outcome'], random_state=66)
    from sklearn.neighbors import KNeighborsClassifier
    training_accuracy = []
    test_accuracy = []
    # setting closest data points or nearest neighbors for testing 
    neighbors_settings = range(1, 11)
    for n_neighbors in neighbors_settings:
        # building the model
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        # training set accuracy
        training_accuracy.append(knn.score(X_train, y_train))
        # test set accuracy
        test_accuracy.append(knn.score(X_test, y_test))

    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(X_train, y_train)

    predictions = knn.predict(diabetes.loc[:, diabetes.columns != 'Outcome'])
    file_name = os.path.splitext('diabetes.csv')[0]
    for idx, prediction in enumerate(predictions):
        if prediction == 1:
            argtrue = "Say in a continuous paragraph with no markdown formatting and assume that this person has diabetes: This person likely has diabetes. After saying that, Suggest a plan they could take to reduce risk factors. Mention diet regarding BMI, glucose, calcium, iron, exercise, and mental health. Remind the patient to consult their doctor for further medical advice. Being the paragraph with what sounds like a definitive diagnosis. Also pretend that this paragraph is a doctor's report to a patient."
            printToGemini(argtrue)
        else:
            argtrue = "Say in a continuous paragraph with no markdown formatting and assume that this person does not have diabetes:  After saying that, suggest a plan maintains good health practices, involving physical, mental, and emotional health regarding preventing diabetes. Remind the patient to consult their doctor for further medical advice. Begin the paragraph with what sounds like a definitive diagnosis. Also pretend that this paragraph is a doctor's report to a patient."
            printToGemini(argtrue)
    os.remove('diabetes.csv')

def getKidneyData():
    data = pd.read_csv('datakd.csv')
    kidney = pd.read_csv('kidney.csv')

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data.loc[:, data.columns != 'Outcome'], data['Outcome'], stratify=data['Outcome'], random_state=66)
    from sklearn.neighbors import KNeighborsClassifier
    training_accuracy = []
    test_accuracy = []
    # setting closest data points or nearest neighbors for testing 
    neighbors_settings = range(1, 11)
    for n_neighbors in neighbors_settings:
        # building the model
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        # training set accuracy
        training_accuracy.append(knn.score(X_train, y_train))
        # test set accuracy
        test_accuracy.append(knn.score(X_test, y_test))

    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(X_train, y_train)

    predictions = knn.predict(kidney.loc[:, kidney.columns != 'Outcome'])
    file_name = os.path.splitext('kidney.csv')[0]
    for idx, prediction in enumerate(predictions):
        if prediction == 1:
            argtrue = "Say in a continuous paragraph with no markdown formatting and assume that this person likely has kidney complications. After saying that, Suggest a plan they could take to reduce risk factors. Mention diet regarding BMI, electrolyte levels, serum creatine, blood urea nitrogen, exercise, and mental health. Remind the patient to consult their doctor for further medical advice. Begin the paragraph with what sounds like a definitive diagnosis. Also pretend that this paragraph is a doctor's report to a patient."
            printToGemini(argtrue)
        else:
            argtrue = "Say in a continuous paragraph with no markdown formatting and assume that this person does not have kidney complications. After saying that, Suggest a plan they could take to reduce risk factors. Mention diet regarding BMI, electrolyte levels, serum creatine, blood urea nitrogen, exercise, and mental health. Remind the patient to consult their doctor for further medical advice. Begin the paragraph with what sounds like a definitive diagnosis. Also pretend that this paragraph is a doctor's report to a patient."
            printToGemini(argtrue)
    os.remove('kidney.csv')


def getCancerData():
    data = pd.read_csv('datac.csv')
    cancer = pd.read_csv('cancer.csv')

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data.loc[:, data.columns != 'Outcome'], data['Outcome'], stratify=data['Outcome'], random_state=66)
    from sklearn.neighbors import KNeighborsClassifier
    training_accuracy = []
    test_accuracy = []
    # setting closest data points or nearest neighbors for testing 
    neighbors_settings = range(1, 11)
    for n_neighbors in neighbors_settings:
        # building the model
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        # training set accuracy
        training_accuracy.append(knn.score(X_train, y_train))
        # test set accuracy
        test_accuracy.append(knn.score(X_test, y_test))

    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(X_train, y_train)

    predictions = knn.predict(cancer.loc[:, cancer.columns != 'Outcome'])
    file_name = os.path.splitext('cancer.csv')[0]
    for idx, prediction in enumerate(predictions):
        if prediction == 1:
            argtrue = "Say in a continuous paragraph with no markdown formatting and assume that this person has cancer markers After saying that, clarify that while it is just a risk it is not something that should be taken lightly. If caught early by medical professionals, it can be very treatable. Encourage the patient to seek further, more advanced medical help. Begin the paragraph with a diagnosis that gives a slight sense of urgency. Also pretend that this paragraph is a doctor's report to a patient."
            printToGemini(argtrue)
        else:
            argtrue = "Say in a continuous paragraph with no markdown formatting and assume that this person does not have cancer markers. After saying that, clarify that while nothing is present right now, this does not signify a cancer-free future. If caught early by medical professionals, it can be very treatable. Encourage the patient maintain a healthy lifestyle that would help prevent cancer development, with a disclaimer that it is not a foolproof plan and randomness really does happen. Begin the paragraph with a clear diagnosis statement. Also pretend that this paragraph is a doctor's report to a patient."
            printToGemini(argtrue)
    os.remove('cancer.csv')

def write_person_data_to_csv(people_data, filename):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:  # Use 'a' mode to append to existing file
        csv_writer = csv.writer(csvfile)

        # Write headers if the file doesn't exist
        if not file_exists:
            headers = [key for key, _ in people_data[0]]
            csv_writer.writerow(headers)

        # Write data for each person
        for person in people_data:
            data = [value for _, value in person]
            csv_writer.writerow(data)



app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sign')
def sign():
    return render_template('sign.html')

@app.route('/framework', methods=['GET', 'POST'])
def framework():
    return render_template('framework.html')

@app.route('/report', methods=['GET', 'POST'])
def report():
    global genText
    return_text = format_markdown(genText)
    return render_template('report.html', generated_text=return_text)

@app.route('/processdbinput', methods=['GET', 'POST'])
def processdbinput():
    if request.method == 'POST':
        age_input = request.form['age']
        bmi_input = request.form['bmi']
        glucose_input = request.form['glucose']
        iron_input = request.form['iron']
        calcium_input = request.form['calcium']
        def get_person_data_from_input():
            age = age_input
            bmi = bmi_input
            glucose = glucose_input
            iron = iron_input
            calcium = calcium_input
            return [('Age', age), ('BMI', bmi), ('Glucose', glucose), ('Iron', iron), ('Calcium', calcium), ('Outcome', 0)]
        patientdbdata = [get_person_data_from_input()]
        write_person_data_to_csv(patientdbdata, 'diabetes.csv')
        getDiabetesData()
        
    return report()

@app.route('/processkidneyinput', methods=['GET', 'POST'])
def processkidneyinput():
    if request.method == 'POST':
        age_input = request.form['age']
        bmi_input = request.form['bmi']
        electrolyte_input = request.form['electrolyte']
        creatine_input = request.form['creatine']
        nitrogen_input = request.form['nitrogen']
        def get_person_data_from_input():
            age = age_input
            bmi = bmi_input
            electrolyte = electrolyte_input
            creatine = creatine_input
            nitrogen = nitrogen_input
            return [('Age', age), ('BMI', bmi), ('Electrolyte', electrolyte), ('Creatine', creatine), ('Nitrogen', nitrogen), ('Outcome', 0)]
        patientdbdata = [get_person_data_from_input()]
        write_person_data_to_csv(patientdbdata, 'kidney.csv')
        getKidneyData()
    return report()

@app.route('/processcancerinput', methods=['GET', 'POST'])
def processcancerinput():
    if request.method == 'POST':
        age_input = request.form['age']
        bmi_input = request.form['bmi']
        whitebloodcell_input = request.form['wbc']
        cea_input = request.form['cea']
        af_input = request.form['af']
        def get_person_data_from_input():
            age = age_input
            bmi = bmi_input
            whitebloodcell = whitebloodcell_input
            cea = cea_input
            af = af_input
            return [('Age', age), ('BMI', bmi), ('Whiteblood', whitebloodcell), ('Cea', cea), ('AF', af), ('Outcome', 0)]
        patientdbdata = [get_person_data_from_input()]
        write_person_data_to_csv(patientdbdata, 'cancer.csv')
        getCancerData()
    return report()

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    username = request.form['username']
    password = request.form['password']
    global currentUser
    user = User.query.filter_by(username=username, password=password).first()

    if user:
        currentUser = username
        return render_template('index.html')
    else:
        return render_template('sign.html', error="Try Again: Invalid username or password")

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    username = request.form['username']
    password = request.form['password']
    email = request.form['email']
    global currentUser
    new_user = User(username=username, password=password, email=email)

    try:
        currentUser = username
        db.session.add(new_user)
        db.session.commit()
        create_csv_file(username+".csv")
        return render_template('index.html')
    except IntegrityError:
        db.session.rollback()
        error_message = "Username or email already exists. Please choose a different one."
        return render_template('sign.html', error=error_message)    
    
@app.route('/add_patient', methods=['GET', 'POST'])
def addPatient():
    if request.method == 'POST':
        # Assuming the current user is stored in a variable called currentUser
        # Open the current user's CSV file
        csv_file = f"{currentUser}.csv"
        fieldnames = ['patient', 'latest_report']
        
        # Determine the next incremental number
        next_patient_number = 1
        try:
            with open(csv_file, 'r', newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    next_patient_number = int(row['patient']) + 1
        except FileNotFoundError:
            # If the file doesn't exist, create a new one
            with open(csv_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

        # Append the new data to the file
        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'patient': next_patient_number, 'latest_report': format_markdown(genText)})
            
        return render_template('framework.html')
    return "Operation Failed"
    
@app.route('/add_to_existing_patient', methods=['GET', 'POST'])
def addToExisting():
    stringy1 = format_markdown(genText)
    fileFromField = f"{currentUser}.csv"
    if request.method == 'POST':
        input_number = request.form['ID']  # Assuming input_number is submitted via the form
        s = format_markdown(genText)  # Assuming h is submitted via the form        
        if input_number and s and currentUser:
            fileFromField = f"{currentUser}.csv"
            
            # Read the CSV file and update if necessary
            with open(fileFromField, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                rows = list(reader)
                found = False
                for row in rows:
                    if row[0] == input_number:
                        row[1] = stringy1
                        found = True
                        break  # No need to continue searching
            if found:
                # Rewrite the updated rows back to the CSV file
                with open(fileFromField, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(rows)
                    
                return render_template('framework.html')
            else:
                return render_template('index.html')
        else:
            return "Missing input data."
    else:
        return "Method not allowed."

@app.route('/export_report', methods=['GET', 'POST'])
def export():
    text = format_markdown(genText)

    words = text.split()
    eight_word_elements = []
    for i in range(0, len(words), 9):
        eight_words = ' '.join(words[i:i+9])
        eight_word_elements.append(eight_words)

    response = make_response()

    response.headers['Content-Type'] = 'application/pdf'

    response.headers['Content-Disposition'] = 'attachment; filename="report.pdf"'

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    text_width = 6.5 * inch  
    text_height = 9 * inch   

    x = (letter[0] - text_width) / 2
    y = (letter[1] - text_height) / 2

    c.rect(x, y, text_width, text_height)

    text = c.beginText(x + 10, y + text_height - 20)

    text.setFont("Helvetica", 12)  

    for line in eight_word_elements:
        text.textLine(line)

    c.drawText(text)

    c.showPage()
    c.save()

    response.data = buffer.getvalue()

    buffer.close()

    return response
if __name__ == "__main__":
    app.run(debug=True, port=4999)