# The Perfect Home Finder

The Perfect Home Finder is a data-driven web application designed to help prospective homebuyers find their ideal homes. By analyzing diverse factors such as crime rates, school ratings, neighborhood quality, and housing history, it provides tailored and insightful recommendations using advanced technologies like APIs and Natural Language Processing (NLP).

## Table of Contents
1. [Features](#features)
2. [Technologies Used](#technologies-used)
3. [Installation and Setup](#installation-and-setup)
    - [Setting Up the Environment](#setting-up-the-environment)
    - [Running the Application](#running-the-application)
4. [APIs and Data Sources](#apis-and-data-sources)
5. [Testing](#testing)
6. [Upcoming Features](#upcoming-features)
7. [Contributing](#contributing)
8. [Resources](#resources)
9. [Credits](#credits)

---

## Features

- **Data Mining**: Collects and processes real estate data, crime statistics, school ratings, and neighborhood quality.
- **API Integration**: Fetches real-time data on:
  - Crime rates (with maps)
  - School performance
  - Neighborhood quality
  - Housing Information (prices, bathroom, bedroom, history of homes, etc.)
  - Market insights
  - Geography and maps
  - Environmental data
  - Local Amenitites
  - Demographics
  - Environmental Data
  - Transportation
  - Legal and Regulatory Information
  - Real Estate Agents

- **NLP Integration**: Users can ask natural language queries to refine home searches.
- **Customizable Filters**: Search homes based on location, price range, amenities, and user-defined criteria.
- **Visualization**: Generates visual insights (charts and graphs) to help users make informed decisions.
- **Housing Predictions**: Uses ARIMA/SARIMA and Monte Carlo simulations for housing price forecasts.
- **Interactive Chatbox**: Users can interact with a chatbot for queries about neighborhoods, schools, and properties.

---

## Technologies Used

### **Backend**
- **Python** (3.11)
- **Flask** (web server framework)
- **Transformers** (Hugging Face NLP models)
- **SymPy** (symbolic computation)
- **Pandas** and **NumPy** (data manipulation)

### **Frontend**
- **HTML5**, **CSS3**, and **JavaScript**
- **Bootstrap** (UI framework)
- **GLightbox** (lightbox for images)

### **Data and Visualization**
- **Matplotlib** for charts and graphs
- **Statsmodels** for predictive modeling

### **APIs**
- Crime Data API
- School Ratings API
- Google Maps API

### **Database**
- SQLite or PostgreSQL

---

## Installation and Setup

Follow these steps to set up and run the project.

### Setting Up the Environment

#### **Using Conda**
```bash
# Create and activate environment
conda create --name HouseFinder python=3.11.11
conda activate HouseFinder

# Install dependencies
uv pip install -r requirements.txt
```

#### **Using Virtual Environment (venv)**
```bash
# Install uv for faster installation
pip install uv

# Create virtual environment
uv venv

# Activate virtual environment (Windows)
.\venv\Scripts\activate


# If activation fails on Windows, run the following:
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# For Mac/Linux
source venv/bin/activate


# To deactivate the environment
 deactivate

# If you need to delete the virtual environment:
# Windows
Remove-Item -Recurse -Force .\.venv\

# Mac/Linux
rm -rf .\.venv\

# Install dependencies
uv pip install -r requirements.txt
```

### Dependencies
If `requirements.txt` fails, install libraries individually:
Add these versions to your `requirements.txt`:
```txt
datasets==3.2.0
Flask==3.1.0
matplotlib==3.10.0
Requests==2.32.3
sympy==1.13.1
torch==2.5.1
transformers==4.47.0
numpy==2.2.0
pandas==2.2.3
```

### Run the Application

```bash
# Run the Flask app
python app.py
```

Access the application at: **http://localhost:5000**

---
### OPENLLM Hello
```bash
# Install CUDA
# Download CUDA toolkit from:
https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local

# Verify CUDA installation
nvcc --version

# Install NVIDIA packages
pip install nvidia-pyindex
pip install nvidia-nccl

# Install PyTorch for CUDA
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

I had the RTX 4070 for these settings.

---

## APIs and Data Sources

### **APIs**
1. Crime Data API
2. School Ratings API
3. Google Maps API
4. Zillow Housing Data API

### **Data Sources**
- **Crime Data**: [Kaggle Crime Dataset](https://www.kaggle.com/datasets/taruntiwarihp/crime-world)
- **School Data**: [US Schools Dataset](https://www.kaggle.com/datasets/andrewmvd/us-schools-dataset)
- **Housing Data**: [Zillow Housing Price Data](https://www.kaggle.com/datasets/paultimothymooney/zillow-house-price-data)

---

## Testing

### **NLP Testing**
- Use `testnlp.py` to verify NLP model accuracy.
- Use `testnlpDialo.py` for DialogGPT chatbot testing.

### **Run Chat**
```bash
python app.py
```
Use the interactive chatbox on the front-end to ask questions.

---

## Upcoming Features

- **AI Predictive Analytics**: Enhancing housing price predictions with machine learning models.
- **Mortgage Calculators**: Integrating financial tools for affordability analysis.
- **Enhanced Visualization**: Real-time maps, charts, and graphs for property data.
- **Expanded APIs**: Integration with real estate platforms like Zillow, Realtor.com.

---

## Contributing

We welcome contributions to Perfect Home Finder!
1. Fork the repository.
2. Create a new branch: `git checkout -b feature/your-feature`.
3. Commit your changes: `git commit -m "Add new feature"`.
4. Push the branch: `git push origin feature/your-feature`.
5. Submit a pull request.

---

## Resources

### **Kaggle Datasets**
- [Denver Crime Data](https://www.kaggle.com/datasets/paultimothymooney/denver-crime-data)
- [US Schools Dataset](https://www.kaggle.com/datasets/andrewmvd/us-schools-dataset)

### **APIs and Tools**
- [Google Maps API](https://developers.google.com/maps/)
- [Zillow Housing API](https://www.zillow.com/research/data/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/en/index)
- [OpenLLM](https://pypi.org/project/openllm/)

---

## Credits

**Developers**:
- **Yuri Fung**: Full-Stack Developer, API Integration, NLP, and LLMs [LinkedIn](https://www.linkedin.com/in/yuri-m-fung/)
- **Henry Miller**: Data Analysis, Dataset Management [LinkedIn](https://www.linkedin.com/in/henrymmiller/)

**Special Thanks**:
- [HuggingFace](https://huggingface.co)
- [Kaggle Community](https://www.kaggle.com)
- [Google Maps API](https://developers.google.com/maps/)

---

**Perfect Home Finder** © 2024. All Rights Reserved.



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





# For Yuri:
## Create article page:

### Add news, findings, today's market price, etc
## Create integrated API, with google maps based on search/address obtained from Zillow/school datasets, and find correlation fo the addresses based on user query

## Create database to find the related querys, and stuff so we can improve on this in the future


## Fix Nav
### Design a better one or fix it ig


## Add more photos/make it more appealing for Home.html

## Resize photos in READ.ME