# Halifax Transit Delay & Analytics Dashboard
A machine learning-powered web application that predicts schedule-based delays for the Halifax Transit bus system. Built with an XGBoost regression pipeline, interactive multi-route comparison tools, weather sensitivity simulators, and feature explainability (Tree SHAP).

---

## Link to the Website

* There will be a website link here

---

## Key Features

### 1. Delay Prediction
* Query and compare delay predictions across up to 4 routes at a time
* Displays an 80% confidence interval (the time interval that is wide enough for the model to have 80% confidence)
* Displays heatmaps and line charts to represent this data for predctions

### 2. Model Explanations and Weather Sensitivity Simulations
* Uses Tree SHAP on a horizontal bar plot to display what features are impacting the model the most (time of day, temperature, etc.)
* Because it is difficult to predict weather and precipiation months ahead, the website offers what the delay would look like assuming low precipitation (default), medium precipitation, and high precipitation


### 3. Historical Data Exploration
* Evaluates historical delay distributions and calculate On-Time Performance (trips $\le$ 3 mins delay)
* Offers visualization is the form of graphs across a variety of metrics such a hours and various weather metrics.

---

## How This Was Made

* **Web Framework:** [Streamlit](https://streamlit.io/)
* **Machine Learning:** [XGBoost](https://xgboost.readthedocs.io/) (Gradient Boosted Decision Trees)
* **Data Processing & Storage:** Pandas, NumPy, PyArrow (Apache Parquet)
* **Data Visualization:** Plotly Express
* **Deployment:** GitHub & Streamlit Community Cloud

---

## Local Setup & Execution

If you want to run this application locally on your machine:

### 1. Clone the Repository
Download a copy of the project source code to your computer:
\`\`\`bash
git clone [https://github.com/joelmadadha/HRMTransitWebsite.git](https://github.com/joelmadadha/HRMTransitWebsite.git)
cd HRMTransitWebsite
\`\`\`

### 2. Set Up a Virtual Environment *(Optional, but Recommended)*
To prevent package conflicts with your global Python installation:
\`\`\`bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Mac/Linux
python3 -m venv venv
source venv/bin/activate
\`\`\`

### 3. Install Dependencies
Install all required libraries specified in the deployment recipe:
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### 4. Run the Streamlit Dashboard
Launch the local web server:
\`\`\`bash
streamlit run website.py
\`\`\`

---

## Contact Information
* **Name:** Joel Madadha
* **Email:** joelmadadha@gmail.com
