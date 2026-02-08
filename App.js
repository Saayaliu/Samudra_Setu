import React, { useState } from 'react';
import './App.css';

// Component to display a single prediction result row
const PredictionResultItem = ({ label, prediction, confidence }) => (
  <div className="result-item">
    <div className="result-label">{label}</div>
    <div className="result-value">
      <div className="prediction-text">{prediction}</div>
      <div className="confidence-text">
        Confidence: {confidence ? `${(confidence * 100).toFixed(2)}%` : 'N/A'}
      </div>
    </div>
  </div>
);

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [predictionResults, setPredictionResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Check if file is an image (optional, but good practice)
      if (!file.type.startsWith('image/')) {
        setErrorMessage("Please select a valid image file.");
        setSelectedFile(null);
        setPreviewUrl(null);
        return;
      }
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPredictionResults(null);
      setErrorMessage(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setErrorMessage("Please select an image file first.");
      return;
    }

    setIsLoading(true);
    setPredictionResults(null);
    setErrorMessage(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // The API is expected to return a list of 4 results
      const response = await fetch('http://localhost:8000/predict/', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        // Handle server-side FastAPI errors (like 500s or 400s)
        const detail = data.detail || "An unexpected error occurred on the server.";
        throw new Error(`Prediction failed: ${detail}`);
      }

      // Check for the NaN case (which often results in a valid 200 response with bad data)
      const hasNaN = data.some(item => isNaN(item.confidence));
      if (hasNaN) {
        throw new Error("Prediction returned invalid values (NaN). This usually indicates a model or input image issue.");
      }

      setPredictionResults(data);
    } catch (error) {
      console.error("Prediction failed:", error);
      setErrorMessage(error.message || "Prediction failed. Please check the API server and try again.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="App">
      <div className="header">
        <h1>Otolith Species Classification</h1>
        <p>Analyze otolith images to predict taxonomic and specific epithet details using a multi-output CNN model.</p>
      </div>

      <div className="main-content">
        {/* Left Pane: Upload and Action */}
        <div className="upload-pane">
          <input
            id="fileInput"
            type="file"
            accept="image/*"
            onChange={handleFileChange}
          />
          <div 
            className="upload-box" 
            onClick={() => document.getElementById('fileInput').click()}
            title="Click to upload an image"
          >
            {/* Using a simple SVG icon for file upload */}
            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-8 h-8" style={{ width: '32px', height: '32px', color: 'var(--color-primary)' }}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5m-13.5-9L12 3m0 0 4.5 4.5M12 3v13.5" />
            </svg>
            <p className="upload-box-text">
              {selectedFile ? `File: ${selectedFile.name}` : "Drop or Click to Upload Otolith Image"}
            </p>
          </div>

          {previewUrl && (
            <img src={previewUrl} alt="Otolith Preview" className="image-preview" />
          )}

          <button
            onClick={handlePredict}
            disabled={isLoading || !selectedFile}
            className="predict-button"
          >
            {isLoading ? 'Processing Image...' : 'Run Prediction'}
          </button>
        </div>

        {/* Right Pane: Results */}
        <div className="results-pane">
          <h2>Prediction Results</h2>

          {errorMessage && (
            <div style={{ padding: '15px', backgroundColor: '#fee2e2', color: '#b91c1c', borderRadius: '6px', marginBottom: '20px', fontWeight: 'bold' }}>
              Error: {errorMessage}
            </div>
          )}

          {isLoading && !errorMessage && (
            <div className="loading-message">Loading...</div>
          )}

          {predictionResults && !isLoading && (
            <div className="results-list">
              {predictionResults.map((item, index) => (
                <PredictionResultItem
                  key={index}
                  label={item.label}
                  prediction={item.prediction}
                  confidence={item.confidence}
                />
              ))}
            </div>
          )}

          {!predictionResults && !isLoading && !errorMessage && (
            <div className="loading-message" style={{ color: 'var(--color-text-subtle)' }}>
              Upload an image and press "Run Prediction" to view results.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
