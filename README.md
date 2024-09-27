# The Perfect Home Finder

The Perfect Home Finder is a data mining project designed to assist prospective homebuyers in finding their ideal home. By analyzing various factors such as crime rates, neighborhood quality, school ratings, quality of life, and proximity to landfills, this tool provides a comprehensive overview of potential homes based on user preferences. Leveraging APIs and natural language processing (NLP), it aims to deliver tailored and insightful recommendations.

## Features

- **Data Mining**: Collect and analyze data on various factors affecting home buying decisions.
- **API Integration**: Utilize external APIs to gather real-time data on crime rates, school ratings, neighborhood quality, geography, maps, crime map, housing information (price, bathroom, bedroom, etc), history of home if there is any, local amenities, Demographics, enviromental data, transportation, market options, legal and regulatory information, real-estate agent, market insights, integration with morgage calculators.... So far. we will add more.
- **Natural Language Processing**: Enable users to interact with the system using natural language queries to refine their home search.
- **Customizable Filters**: Allow users to set specific criteria based on their preferences to receive tailored results.
- **Visualization**: Provide visual insights into data to help users make informed decisions.

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
    - `requests` for API calls
    - `pandas` for data manipulation
    - `numpy` for numerical analysis
    - `flask-transformers` for natural language processing
    - `matplotlib` or `seaborn` for data visualization
- **APIs**:
    - Crime data API
    - School ratings API
    - Neighborhood quality API
- **Database**: SQLite or PostgreSQL for storing user preferences and retrieved data









## How to Run

To run the Perfect Home Finder application, follow these steps:

1. **Install Dependencies**:
    First install uv since it will make pip install much faster for requirements.txt
    ```sh
    pip install uv
    ```
    Then create a venv through:
    ```sh
    uv venv
    ```
    Use:
    ```sh
    .\venv\Scripts\activate
    ```
    You may need to run:
    ```
    Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process
    ```
    In order to authenticate....
    To activate your virtual env.

    To deactivate:
    ```sh
    deactivate
    ```
    If by chance you need to delete your venv:
    - For Windows:
    ```sh
    Remove-Item -Recurse -Force .\.venv\
    ```
    - For Mac OSX:
    ```sh
    python rm -rf .\.venv\
    ```


    Dependencies installed invididually:
    ```sh
    pip install flask
    pip install transformers #May need to enable Long Path for Windows
    pip install datasets
    pip install sympy
    pip install -r https://raw.githubusercontent.com/intro-stat-learning/ISLP_labs/v2/requirements.txt 
    ```
    Below are all installation of https://raw.githubusercontent.com/intro-stat-learning/#ISLP_labs/v2/requirements.txt, however if it doesn't work plz try installing individually
    ```sh
    pip install pandas
    pip install ISLP
    pip install numpy
    pip install lxml
    pip install scikit-learn
    pip install joblib
    pip install statsmodels
    pip install lifelines
    pip install pygam
    pip install torch
    pip install pytorch_lightning
    pip install torchmetrics
    pip install torchvision
    pip install matplotlib
    ```
    Installing Deep Learning Frameworks::::
    ```sh
    pip install torchinfo
    pip install torch torchvision torchaudio
    pip install tensorflow
    pip install openllm
    ```
2. **Run the Application**:
    ```sh
    python app.py
    ```

### Additional Information

- **Libraries Used**:
    - `matplotlib` for data visualization
    - `huggingface` for datasets

- **NLP Testing**:
    - Use `testnlp.py` for natural language processing testing.
    - Use `testnlpDialo.py` for Dialo Gpt testing
- **Development Server**:
    - Launch the development server using `python app.py`.



###Resources from Kaggle:
- https://www.kaggle.com/datasets/taruntiwarihp/crime-world
- https://www.kaggle.com/datasets/paultimothymooney/denver-crime-data/versions/457?resource=download

Credits to this guy:
- https://www.kaggle.com/paultimothymooney/datasets?page=5


Resources we haven't used:
- https://www.kaggle.com/datasets/chicago/chicago-crime
- https://www.kaggle.com/datasets/theworldbank/world-bank-intl-education
- https://www.kaggle.com/datasets/LondonDataStore/london-crime
- https://www.kaggle.com/datasets/census/census-bureau-usa
- https://www.kaggle.com/datasets/paultimothymooney/zillow-house-price-data
- https://www.kaggle.com/datasets/datasf/san-francisco
- https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset

- https://www.crcv.ucf.edu/projects/real-world/



For schools:
- https://www.kaggle.com/datasets/andrewmvd/us-schools-dataset
- https://www.kaggle.com/datasets/leomartinelli/bullying-in-schools
- https://www.kaggle.com/datasets/sahirmaharajj/college-exam-results-sat
- https://www.kaggle.com/datasets/noriuk/us-educational-finances
- https://www.kaggle.com/datasets/wsj/college-salaries
- https://www.kaggle.com/datasets/sahirmaharajj/school-student-daily-attendance
- https://www.kaggle.com/datasets/rkiattisak/student-performance-in-mathematics
- https://www.kaggle.com/datasets/pantanjali/unemployment-dataset
- https://www.kaggle.com/datasets/joebeachcapital/school-shootings
- https://www.kaggle.com/datasets/noriuk/us-education-datasets-unification-project
- https://www.kaggle.com/code/zikazika/analysis-of-world-crime



- HuggingFace:
https://huggingface.co/docs/transformers/en/index
https://huggingface.co/transformers/v3.5.1/installation.html
https://huggingface.co/microsoft/DialoGPT-medium
https://huggingface.co/deepset/roberta-base-squad2
- https://haystack.deepset.ai/tutorials/01_basic_qa_pipeline


For maps:
https://www.openstreetmap.org/export#map=15/40.56215/-105.06626
https://crimegrade.org/
- https://developers.google.com/maps/


- https://crimegrade.org/safest-places-in-denver-co/
https://www.zillow.com/
https://developer.schooldigger.com/#plans
- https://collegescorecard.ed.gov/data/
- https://nces.ed.gov/ipeds/use-the-data/
- https://andyreiter.com/datasets/
- https://github.com/eci-io/climategpt-evaluation
- https://arxiv.org/abs/2401.09646
- https://huggingface.co/datasets/eci-io/climate-evaluation




Python:
- https://pypi.org/project/openllm/
- https://www.llama.com/llama3/license/




- https://ralphieslist.colorado.edu/housing?bounds=40.01904,39.99459,-105.24651,-105.28277
Something like this would be cool:


Would be cool to write a data-to-paper kinda thing for this as well:
- https://arxiv.org/pdf/2404.17605


API:
https://www.zillow.com/research/data/


https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety