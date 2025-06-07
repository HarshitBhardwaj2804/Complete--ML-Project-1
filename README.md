## My first End to End machine learning project.

## Structure of the project. 
    ML-Project/                             # Main folder
    │
    ├── artifact/
    │   └── raw_data.csv
    │   └── test_data.csv
    |   └── train_data.csv
    |   └── model.pkl                       # Final trained model 
    |   └── preprocessor.pkl                # Object which handles data preprocessing
    |
    ├── logs/                               # Contains all the logs from the project
    │
    ├── notebook/          
    │   └── data/
    |       └── data.csv                    # main data file
    |   └── EDA.ipynb  
    |
    |── src/ 
    |   └── components/
    |       └── data_ingestion.py           # File responsible to full data from multiple sources
    |       └── data_transformation.py      # Python file to do preprocessing task on the raw data.
    |       └── model_trainer.py            # Python file to train and select best model.
    |   └── pipeline/
    |       └── prediction_pipeline.py      # python file takes the new data and preforms prediction.
    |       └── train_pipeline.py
    |   └── exception.py                    # Python file to raise custom exceptions
    |   └── utils.py                        
    |   └── logger.py
    |
    ├── templates/
    │   └── index.html
    |   └── home.html
    |   └── result.html       
    │
    ├── app.py                              # Main flask file
    ├── setup.py
    ├── requirements.txt
    ├── README.md
    └── .gitignore

## Project link:
    https://student-perfomance-prediction.onrender.com
