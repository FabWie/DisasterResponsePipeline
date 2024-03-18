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

# Guidelines for executing data processing, model fitting, and launching the web application :
Run these commands in the project's root directory to set up your database and model
- ETL pipeline cmds: python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
- ML pipeline cmds: python models/train_classifier.py ../data/DisasterResponse.db models/classifier.pkl
- Web app cmds: python run.py => click on http://127.0.0.1:3000

# Screenshoots:
<img width="881" alt="image" src="https://github.com/FabWie/DisasterResponsePipeline/assets/30210533/572cb3e1-01d7-4894-b836-8bfae8a3c216">


# Data Scoure:
Udacity "Data Science" Nanodegree program / Appen (formerly Figure 8)

# Acknowledgements :
Thanks to Appen (formerly Figure 8) for providing the data.
