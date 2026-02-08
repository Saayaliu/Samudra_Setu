Otolith Species Prediction Web Application
This project implements a multi-output Convolutional Neural Network (CNN) model to predict the taxonomic details (Scientific Name, Family, Genus, Specific Epithet) of fish based on otolith (ear stone) images. The application is built with a decoupled architecture:

Backend (API): Python, FastAPI, and TensorFlow

Frontend (GUI): React and CSS

🚀 Getting Started
Prerequisites
Docker: (Recommended for easy deployment)

Python 3.10+ (If running locally)

Node.js & npm (If running the React frontend locally)

Project Structure
.
├── api/                   # FastAPI Backend & Model
│   ├── main.py            # API logic, prediction, and class mapping
│   ├── otolith_model.keras # **Model File (MUST BE PRESENT)**
│   ├── requirements.txt   # Python dependencies (TensorFlow, FastAPI)
│   └── Dockerfile         # Docker build instructions
├── gui/                   # React Frontend
│   ├── src/
│   │   ├── App.js         # React UI components and API interaction
│   │   └── App.css        # Styling for the dark, modern theme
│   └── package.json       # Node.js dependencies
└── .gitignore             # Files to ignore (e.g., node_modules, venv)

⚙️ Local Development (Without Docker)
1. Backend Setup
Navigate to the api directory: cd api

Create and activate a virtual environment:

Windows: python -m venv venv then venv\Scripts\activate

macOS/Linux: python3 -m venv venv then source venv/bin/activate

Install Python requirements: pip install -r requirements.txt

Start the API server: uvicorn main:app --reload
The API will run on http://localhost:8000.

2. Frontend Setup
Open a new terminal and navigate to the gui directory: cd gui

Install Node dependencies: npm install

Start the React server: npm start
The GUI will open in your browser on http://localhost:3000.

🐳 Deployment with Docker (Recommended)
This method only requires Docker installed and running.

Navigate to the api directory: cd api

Build the Docker image: docker build -t otolith-api .

Run the container: docker run -d --name otolith-container -p 8000:8000 otolith-api

Access the API documentation at http://localhost:8000/docs and the application GUI via the separate React server (Step 2 above).