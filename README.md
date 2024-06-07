# MediBuddy

**Recipient of the first-place prize at LA Hacks for the Patient Safety Technology Challenge**

## Inspiration

MediBuddy is inspired by the serious issue of medical errors in healthcare, which leads to approximately 250,000 deaths in the United States each year. Despite extensive training, healthcare professionals still make errors, resulting in incorrect treatment plans or the overlooking of major diseases. Leveraging machine learning models, MediBuddy analyzes symptoms, body measurements, and other inputs to provide accurate, personalized suggestions on diagnoses and treatments based on previous case data. The aim is to reduce misdiagnosis rates by utilizing a large database of medical history to produce accurate predictions, ultimately complementing the work of physicians.

## What it does

MediBuddy is a website powered by a machine learning disease prediction model aimed at solving the problem of diagnostic errors in healthcare. The machine learning model provides confident inferences about whether a patient likely has a certain condition given input from a healthcare professional, acting as a virtual physician’s assistant to help reinforce diagnoses and compile personalized reports of recommendations for patients. These inferences are then fed to Google's Gemini to generate a report that offers recommendations regarding diet, exercise, and other habits to treat or prevent specific conditions.

## How we built it

### Roles:

- **Choidorj Bayarkhuu:** Primary frontend developer (CS @ UCLA)
- **Arnav Roy:** Primary backend logic developer, front end, machine learning (CS @ UCLA)
- **Stanley Sha:** Backend developer and presentation contributor (CS @ UCI)
- **Emma Wu:** Data processing, frontend developer, Gemini Integration, and presentation lead (STATS + DS @ UCLA)

### Frontend:

The frontend of MediBuddy is mostly built using HTML, CSS, with some SCSS and JavaScript. Key components include:
- Landing page providing information about the site and navigation to other pages.
- Login page for user authentication.
- User input page for gathering patient data.
- Results page displaying machine learning predictions and customized reports.

### Backend:

MediBuddy's backend incorporates a K-Nearest Neighbors machine learning model and Gemini LLM via Flask and SQLite. Key components include:
- Training the machine learning model on previous diagnosis data.
- Using Flask to create an app for frontend-backend communication.
- Processing patient data and feeding results to the machine learning model.
- Integrating Google Gemini to generate custom AI reports.

## Libraries Used

### Web Development
- Flask: A lightweight WSGI web application framework in Python.
- Flask_SQLAlchemy: Flask extension for SQLAlchemy, a SQL toolkit and Object-Relational Mapper.
- render_template: Flask function for rendering HTML templates.
- request: Flask module for handling HTTP requests.
- redirect, url_for: Flask functions for URL redirection.
- jsonify: Flask function for creating JSON responses.
- make_response: Flask function for creating custom responses.
- send_file: Flask function for sending files as responses.

### Data Processing and Machine Learning
- pandas: Library providing data structures and data analysis tools for Python.
- numpy: Library for numerical computing in Python.
- StandardScaler: Scikit-learn class for standardizing features by removing the mean and scaling to unit variance.
- DecisionTreeClassifier: Scikit-learn class for decision tree classification.
- MLPClassifier: Scikit-learn class for multi-layer perceptron classifier.

### Date and Time
- datetime: Module providing classes for manipulating dates and times.

### File Handling
- csv: Module providing classes and functions for reading and writing CSV files.
- os: Module providing functions for interacting with the operating system.

### AI and Natural Language Processing
- google.generativeai: Package for accessing Google's Generative AI models.

### Database
- SQLAlchemy: SQL toolkit and Object-Relational Mapper (ORM) for Python.

### PDF Generation
- reportlab: Library for creating PDF documents in Python.
- canvas: Module for drawing 2D graphics on a PDF.
- BytesIO: Module for handling binary data in memory.

### Miscellaneous
- IntegrityError: Exception raised when a database integrity constraint is violated.
- random: Module providing functions for generating random numbers.
- letter: Size specification for US letter paper size.
- inch: Unit of length for specifying dimensions.
- simpleSplit: Function for splitting text into lines for PDF generation.

### Image Gallery
![image](https://github.com/aroy23/MediBuddy/assets/83829580/d4e782fc-8c62-456d-8e15-ca79cf1617eb)
![image](https://github.com/aroy23/MediBuddy/assets/83829580/e08e594d-afa7-46fc-b037-e3ed7456a369)
![image](https://github.com/aroy23/MediBuddy/assets/83829580/0cf96739-6a5f-49eb-87c9-38c6b0fb23ac)
![image](https://github.com/aroy23/MediBuddy/assets/83829580/2208532e-5a16-4bcc-b961-fd43c91f815a)





