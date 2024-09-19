# The Perfect Home Finder

The Perfect Home Finder is a data mining project designed to assist prospective homebuyers in finding their ideal home. By analyzing various factors such as crime rates, neighborhood quality, school ratings, quality of life, and proximity to landfills, this tool provides a comprehensive overview of potential homes based on user preferences. Leveraging APIs and natural language processing (NLP), it aims to deliver tailored and insightful recommendations.

## Features

- **Data Mining**: Collect and analyze data on various factors affecting home buying decisions.
- **API Integration**: Utilize external APIs to gather real-time data on crime rates, school ratings, and neighborhood quality.
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
    ```sh
    pip install flask
    pip install transformers #May need to enable Long Path for Windows
    pip install datasets
    pip install sympy
    pip install -r https://raw.githubusercontent.com/intro-stat-learning/ISLP_labs/v2/requirements.txt #Below are all installation of https://raw.githubusercontent.com/intro-stat-learning/#ISLP_labs/v2/requirements.txt, however if it doesn't work plz try installing individually

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
# Installing Deep Learning Frameworks::::
    pip install torchinfo
    pip install torch torchvision torchaudio
    pip install tensorflow
    ```

2. **Run the Application**:
    ```sh
    python app.py
    ```

### Additional Information

- **Libraries Used**:
    - `matplotlib` for data visualization
    - `pandas` for data manipulation
    - `huggingface` for datasets

- **NLP Testing**:
    - Use `testnlp.py` for natural language processing testing.

- **Development Server**:
    - Launch the development server using `python app.py`.

