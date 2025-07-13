# ANN Stock Predictor Web App

A simple web application that predicts stock prices using an Artificial Neural Network (ANN). The user enters a stock ticker(can find it online), and the app trains a model using the first 80% of the range and showcases the predictions for the last 20% of the range.

## Key Features

*   **User-Friendly Interface**: A clean and simple form to input a stock ticker.
*   **Automatic Data Fetching**: Uses the `yfinance` library to download the last three years of stock data.
*   **On-the-Fly Model Training**: An Artificial Neural Network is built and trained in real-time for each request.
*   **Interactive Visualizations**: Displays interactive plots created with Plotly, which compare prediced prices vs actual prices.
*   **Responsive Design**: The interface is designed to work well on both desktop and mobile devices.
*   **Loading Indicator**: Shows a loading animation while the model is training to improve user experience.

> **Note**: Currently, the training takes 500 epochs, which is time-consuming and also depends on the date range.

## Tech Stack

This project is built with a combination of Python for the backend and standard web technologies for the frontend.

*   **Backend**: Flask, Gunicorn
*   **Machine Learning**: TensorFlow (Keras), Scikit-Learn, Pandas, NumPy
*   **Data Source**: `yfinance`
*   **Plotting**: Plotly
*   **Frontend**: HTML, CSS

## Project Structure

The project is organized into modules for clarity and maintainability.

├── app.py                   # Main Flask application file  
├── predict    
│   ├── data_collect.py      # Functions for data preprocessing  
│   ├── __init__.py          # Makes 'predict' a Python package  
│   └── run_predict.py       # Main prediction and plotting logic  
├── req2.txt                 # enough dependencies to deploy the model  
├── requirements.txt         # Project dependencies for all teh modules  
├── static    
│   ├── main.css             # Styles for the web page  
├── stockml.ipynb    
└── templates    
    ├── base.html    
    └── index.html           # HTML template for the user interface  


### How to Run This Project Locally

To get a local copy up and running, follow these simple steps.

#### Prerequisites

*   Python 3.11
*   `pip` (Python package installer)

#### Installation & Setup

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/Bhuvan-56/stockmlproject.git
    cd stockmlproject
    ```

2.  **Create and activate a virtual environment (recommended):**
    *   On Windows:
        ```sh
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   On macOS & Linux:
        ```sh
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Install the required packages:**
    *   For a full local setup with all original packages, use `requirements.txt`:
        ```sh
        pip install -r requirements.txt
        ```
    *   To use the same lightweight package list as the deployment, use `req2.txt`:
        ```sh
        pip install -r req2.txt
        ```

4.  **Run the Flask application:**
    ```sh
    flask run
    ```
    Alternatively, you can run:
    ```sh
    python app.py
    ```

5.  **Open your browser** and navigate to `http://127.0.0.1:5000` to see the application in action.

#### You can check out the deployed web application
> **Note**: This is deployed on render and may face temporary downtime and takes some time to fire up after temporary inactivity.

**[Deployed on Render](https://the-ann-stock-predictor.onrender.com/)** | **[GitHub Repository](https://github.com/Bhuvan-56/stockmlproject)**