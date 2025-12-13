# SpotTox
# Overview
SpotTox is an ML/AI-powered web application designed to analyze online conversations and detect 
toxic or harmful language. The system allows users to upload conversation threads, 
select a machine learning model, and receive toxicity scores along with visual summaries. 
SpotTox is built to be simple and accessible for general users who want to evaluate toxicity 
without advanced technical knowledge.

# Team
- **Krisha Patel**
- **Atishin Sadaf**
- **Makayla Gunter** 

# Features
- Upload conversation threads in CSV format
- Analyze entire conversation threads for toxicity
- Select from trained machine learning models
- View results through charts and summaries
- Highlight the most toxic messages in a conversation
- Web-based interface with no installation required for users

# System Architecture
SpotTox follows a client–server architecture:
- Frontend: React (JavaScript) for user interaction and visualization
- Backend: Flask (Python) for data processing and API handling
- Machine Learning: PyTorch and Hugging Face Transformers for model training and inference

# Technologies Used
- Frontend: React, JavaScript, HTML, CSS
- Backend: Python, Flask, Flask-CORS
- Machine Learning: PyTorch, Hugging Face Transformers
- NLP Tools: spaCy
- Development Tools: Visual Studio Code, PyCharm, Git

# How It Works
1. The user uploads a CSV file containing a conversation thread.
2. The user selects a trained model from a dropdown menu.
3. The backend preprocesses the text and runs toxicity analysis.
4. The system aggregates toxicity scores and generates results.
5. The frontend displays charts, summaries, and flagged messages.

# Installation and Setup
# Frontend

- npm install
- npm start

# Backend

- git clone https://github.com/atishinsadaf/spottox.git
- cd spottox-backend
- python -m venv venv
- source venv/bin/activate   # Windows: venv\Scripts\activate
- pip install -r requirements.txt
- python app.py


