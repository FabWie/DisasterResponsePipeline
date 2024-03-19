# Disaster Response Pipeline :
"Disaster Response Pipeline" project is part of the Udacity "DataScience" nanodegree program.

# Project Overview :
This project utilizes disaster data sourced from Appen (formerly Figure 8) to construct a model for an API capable of classifying disaster messages.
A web application has been developed, enabling emergency workers to input new messages and receive classification results across multiple categories.
Additionally, the web app provides visualizations of the data.

# Installations :
The project was developed using Python in the PyCharm IDE. It leveraged the following packages:
- flask
- joblib
- json
- numpy
- nltk
- pandas
- pickle
- plotly
- sqlalchemy
- sklearn (scikit-learn)
- sys

# File structure:
The data is structured in three folders (app/data/models):
1. app: <br />
  contains a folder "templates" and a "run.py"
  - 1.1 folder templates: #contains file for web app
  - 1.1.1 go.html
  - 1.1.2 master.html
  - 1.2 run.py  
2. data: <br />
   contains "DisasterResponse.db", "categories.csv", "messages.csv" and the process_data.py
   - 2.1 "DisasterResponse.db"
   - 2.2 "categories.csv"
   - 2.3 "messages.csv"
   - 2.4 process_data.py
3. models: <br />
   contains "classifier.pkl" and train_classifier.py
   - 3.1 contains "classifier.pkl"
   - 3.2 train_classifier.py 

# Guidelines for executing data processing, model fitting, and launching the web application :
Run these commands in the project's root directory to set up your database and model
- ETL pipeline cmds: python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
- ML pipeline cmds: python models/train_classifier.py ../data/DisasterResponse.db models/classifier.pkl
- Web app cmds: python run.py => click on http://127.0.0.1:3000

# Screenshoots:
<img width="881" alt="image" src="https://github.com/FabWie/DisasterResponsePipeline/assets/30210533/572cb3e1-01d7-4894-b836-8bfae8a3c216">
<img width="795" alt="image" src="https://github.com/FabWie/DisasterResponsePipeline/assets/30210533/91eadd1a-b81c-4eb8-9c12-e3e76277c0c6">

Example of Classify Message:

<img width="922" alt="image" src="https://github.com/FabWie/DisasterResponsePipeline/assets/30210533/7fe88574-79e4-4055-883c-edda3fa64aab">
<img width="872" alt="image" src="https://github.com/FabWie/DisasterResponsePipeline/assets/30210533/1c7f2cd2-c1da-4fbd-9025-be66517967cc">
<img width="873" alt="image" src="https://github.com/FabWie/DisasterResponsePipeline/assets/30210533/6282636b-cf29-4039-9fd9-69905d727e32">



# Data Scoure:
Udacity "Data Science" Nanodegree program / Appen (formerly Figure 8)

# Acknowledgements :
Thanks to Appen (formerly Figure 8) for providing the data.
