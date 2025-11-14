import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://localhost:5000/api';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setPrediction(null);
      setError('');
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_BASE}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000, // 30 second timeout
      });
      setPrediction(response.data);
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err.response?.data?.error || 'Prediction failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const formatConfidence = (confidence) => {
    return (confidence * 100).toFixed(2);
  };

  const formatClassName = (className) => {
    return className
      .replace(/__/g, ' - ')
      .replace(/_/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();
  };

  const getWarningClass = (warning) => {
    if (warning.includes('❌')) return 'error-message';
    if (warning.includes('⚠️')) return 'warning-message';
    if (warning.includes('ℹ️')) return 'info-message';
    return 'success-message';
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🌿 Plant Disease Classifier</h1>
        <p>Upload a plant leaf image to identify potential diseases</p>
        <p className="app-subtitle">Supports: Pepper, Potato, and Tomato plants</p>
      </header>

      <main className="main-content">
        <div className="upload-section">
          <div className="file-input-container">
            <input
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="file-input"
              id="file-input"
            />
            <label htmlFor="file-input" className="file-input-label">
              📷 Choose Image
            </label>
            <div className="file-requirements">
              Supported formats: JPG, PNG, GIF, BMP
            </div>
          </div>
          
          {previewUrl && (
            <div className="image-preview">
              <img src={previewUrl} alt="Preview" className="preview-image" />
              <div className="image-info">
                <p>Image ready for analysis</p>
              </div>
            </div>
          )}

          <button 
            onClick={handlePredict} 
            disabled={!selectedFile || loading}
            className={`predict-button ${loading ? 'loading' : ''}`}
          >
            {loading ? (
              <>
                <div className="spinner"></div>
                Analyzing...
              </>
            ) : (
              '🔍 Analyze Image'
            )}
          </button>
        </div>

        {error && (
          <div className="error-message">
            ❌ {error}
          </div>
        )}

        {prediction && (
          <div className="results-section">
            <div className="prediction-card">
              <h2>Analysis Results</h2>
              
              {prediction.warning && (
                <div className={getWarningClass(prediction.warning)}>
                  {prediction.warning}
                </div>
              )}
              
              <div className="prediction-main">
                <div className="confidence-level">
                  <div 
                    className="confidence-bar"
                    style={{ width: `${formatConfidence(prediction.confidence)}%` }}
                  ></div>
                  <span className="confidence-text">
                    Confidence: {formatConfidence(prediction.confidence)}%
                    {!prediction.is_reliable && ' ⚠️'}
                  </span>
                </div>

                <div className="disease-info">
                  <h3>{prediction.disease_name}</h3>
                  <p className="description">{prediction.description}</p>
                  
                  <div className="treatment-advice">
                    <h4>Treatment Recommendations:</h4>
                    <p>{prediction.treatment}</p>
                  </div>
                </div>
              </div>

              <div className="all-predictions">
                <h4>All Possibilities:</h4>
                <div className="predictions-list">
                  {Object.entries(prediction.all_predictions)
                    .sort(([,a], [,b]) => b - a)
                    .slice(0, 5)
                    .map(([className, confidence]) => (
                      <div key={className} className={`prediction-item ${className === prediction.class ? 'top-prediction' : ''}`}>
                        <span className="class-name">
                          {formatClassName(className)}
                          {className === prediction.class && ' 🎯'}
                        </span>
                        <span className="class-confidence">
                          {formatConfidence(confidence)}%
                        </span>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {!prediction && !loading && (
          <div className="info-section">
            <h3>How to use:</h3>
            <ul>
              <li>📷 Upload a clear image of a plant leaf</li>
              <li>🌿 Works best with Pepper, Potato, or Tomato plants</li>
              <li>🔍 Ensure the leaf is well-lit and centered</li>
              <li>⏱️ Analysis typically takes 2-5 seconds</li>
            </ul>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;