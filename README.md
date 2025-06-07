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
    |       └── data_ingestion.py
    |       └── data_transformation.py
    |       └── model_trainer.py
    |   └── pipeline/
    |       └── prediction_pipeline.py
    |       └── train_pipeline.py
    |
    |   └── exception.py
    |   └── utils.py
    |   └── logger.py
    |
    ├── templates/
    │   └── index.html
    |   └── home.html
    |   └── result.html       
    │
    ├── app.py
    ├── setup.py
    ├── requirements.txt
    ├── README.md
    └── .gitignore
