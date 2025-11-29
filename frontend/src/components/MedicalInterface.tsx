import React, { useState, useRef, useCallback } from 'react';
import MedicalResultCard from './MedicalResultCard';
import { processMedicalDocument, MedicalResult, MedicalTestData } from '../services/api';

const MedicalInterface: React.FC = () => {
  const [text, setText] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<MedicalResult | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileSelect = (selectedFile: File) => {
    const validTypes = ['.pdf', '.docx', '.doc', '.txt', '.png', '.jpg', '.jpeg'];
    const fileExt = '.' + selectedFile.name.split('.').pop()?.toLowerCase();
    
    if (!validTypes.includes(fileExt)) {
      setError('Invalid file type. Please upload PDF, DOCX, TXT, or image files.');
      return;
    }
    
    setFile(selectedFile);
    setError(null);
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  }, []);

  const handleProcess = async () => {
    if (!text.trim() && !file) {
      setError('Please enter text or upload a document to process.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await processMedicalDocument(text, file);
      
      // Check if it's a legal document
      if (result.error && result.error.includes('legal')) {
        setError('⚠️ This appears to be a legal document. Please use the Legal Documents tab instead.');
        setResults(null);
      } else if (result.error) {
        setError(result.error);
        setResults(null);
      } else {
        setResults(result);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to process document. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setResults(null);
    setText('');
    setFile(null);
    setError(null);
  };

  return (
    <div className="interface-container">
      <div className="interface-header">
        <h1 className="interface-title">🏥 Medical Report Simplifier</h1>
        <p className="interface-subtitle">
          Upload lab reports or medical documents to understand your health results in simple terms
        </p>
      </div>

      {!results && (
        <div className="upload-section">
          <div 
            className={`upload-area ${dragOver ? 'drag-over' : ''}`}
            onDrop={handleDrop}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="upload-icon">🔬</div>
            <p className="upload-text">
              Drag & drop your medical report here or{' '}
              <span className="browse-button">browse</span>
            </p>
            <p className="upload-hint">Supports PDF, DOCX, TXT, and images of lab reports</p>
            <input
              ref={fileInputRef}
              type="file"
              className="file-input"
              accept=".pdf,.docx,.doc,.txt,.png,.jpg,.jpeg"
              onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
            />
          </div>

          {file && (
            <div className="file-selected">
              <span>📎</span>
              <span className="file-name">{file.name}</span>
              <button className="remove-file" onClick={() => setFile(null)}>×</button>
            </div>
          )}

          <div className="text-input-section">
            <div className="divider">
              <span className="divider-line"></span>
              <span className="divider-text">OR paste your lab results</span>
              <span className="divider-line"></span>
            </div>
            <textarea
              className="text-area"
              placeholder="Paste your medical report or lab results here...

Example:
HEMOGLOBIN: 8.5 g/dL (Reference Range: 12.0-17.0 g/dL)
FASTING BLOOD SUGAR: 180 mg/dL (Reference: 70-100 mg/dL)
SERUM CREATININE: 2.5 mg/dL (Normal: 0.7-1.3 mg/dL)"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </div>

          <button 
            className="process-button" 
            onClick={handleProcess}
            disabled={loading || (!text.trim() && !file)}
          >
            {loading ? (
              <>
                <span className="spinner" style={{ width: 20, height: 20, borderWidth: 2 }}></span>
                Analyzing...
              </>
            ) : (
              <>
                🔍 Analyze Report
              </>
            )}
          </button>
        </div>
      )}

      {error && (
        <div className="error-message">
          <span className="error-icon">⚠️</span>
          <span>{error}</span>
        </div>
      )}

      {loading && !results && (
        <div className="loading">
          <div className="spinner"></div>
          <p className="loading-text">Analyzing your medical report... This may take a moment.</p>
        </div>
      )}

      {results && (
        <div className="results-section">
          <div className="results-header">
            <h2 className="results-title">🩺 Analysis Results</h2>
            <button className="clear-button" onClick={handleClear}>
              ← Analyze Another
            </button>
          </div>

          {/* Summary Card */}
          <div className="summary-card medical">
            <h3 className="summary-title">Report Summary</h3>
            <div className="summary-stats">
              <div className="stat-item">
                <div className="stat-value">{results.summary?.total_tests || results.results?.length || 0}</div>
                <div className="stat-label">Tests Found</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{results.summary?.abnormal_count || 0}</div>
                <div className="stat-label">Abnormal</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{results.summary?.critical_count || 0}</div>
                <div className="stat-label">Critical</div>
              </div>
            </div>
          </div>

          {/* Medical Result Cards */}
          {results.results?.map((test: MedicalTestData, index: number) => (
            <MedicalResultCard key={index} data={test} index={index + 1} />
          ))}

          {/* Recommendations */}
          {results.summary?.recommendations && results.summary.recommendations.length > 0 && (
            <div className="clause-card">
              <div className="clause-header">
                <span className="clause-number">💡 Health Recommendations</span>
              </div>
              <div className="clause-content">
                {results.summary.recommendations.map((rec: string, i: number) => (
                  <p key={i} style={{ marginBottom: '0.5rem' }}>{rec}</p>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default MedicalInterface;
