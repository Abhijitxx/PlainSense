import React, { useState, useRef, useCallback } from 'react';
import ClauseRow from './ClauseRow';
import { processLegalDocument, LegalResult, ClauseData, LawReference, translateText } from '../services/api';

type TranslationLang = 'none' | 'hi' | 'ta';

// Law Popup Component
const LawPopup: React.FC<{ law: LawReference; onClose: () => void }> = ({ law, onClose }) => {
  return (
    <div className="law-popup-overlay" onClick={onClose}>
      <div className="law-popup" onClick={(e) => e.stopPropagation()}>
        <button className="law-popup-close" onClick={onClose}>×</button>
        <h3 className="law-popup-title">📚 {law.law_name}</h3>
        {law.section && <p className="law-popup-section"><strong>Section:</strong> {law.section}</p>}
        {law.article && <p className="law-popup-article"><strong>Article:</strong> {law.article}</p>}
        <div className="law-popup-reference">
          <strong>Reference:</strong> {law.full_reference}
        </div>
        <div className="law-popup-explanation">
          <h4>📖 What this law means:</h4>
          <p>{law.explanation}</p>
        </div>
        {law.context && (
          <div className="law-popup-context">
            <h4>📍 Where it appears in your document:</h4>
            <p className="context-text">...{law.context}...</p>
          </div>
        )}
      </div>
    </div>
  );
};

const LegalInterface: React.FC = () => {
  const [text, setText] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<LegalResult | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Translation state - single dropdown for all clauses
  const [selectedLang, setSelectedLang] = useState<TranslationLang>('none');
  const [translations, setTranslations] = useState<{[clauseIndex: number]: string}>({});
  const [isTranslating, setIsTranslating] = useState(false);
  
  // Law popup state
  const [selectedLaw, setSelectedLaw] = useState<LawReference | null>(null);

  const handleFileSelect = (selectedFile: File) => {
    // Check if it's a valid file type
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
      const result = await processLegalDocument(text, file);
      
      // Check if it's a medical document
      if (result.error && result.error.includes('medical')) {
        setError('⚠️ This appears to be a medical document. Please use the Medical Documents tab instead.');
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
    setSelectedLang('none');
    setTranslations({});
    setSelectedLaw(null);
  };

  // Handle global translation for all clauses
  const handleTranslationChange = async (lang: TranslationLang) => {
    setSelectedLang(lang);
    
    if (lang === 'none') {
      setTranslations({});
      return;
    }
    
    if (!results?.clauses || results.clauses.length === 0) return;
    
    setIsTranslating(true);
    const newTranslations: {[key: number]: string} = {};
    
    try {
      // Translate all clauses
      for (let i = 0; i < results.clauses.length; i++) {
        const clause = results.clauses[i];
        const textToTranslate = clause.english?.plain || clause.simplified || '';
        
        if (textToTranslate) {
          try {
            const result = await translateText(textToTranslate, lang);
            newTranslations[i] = result.translation || 'Translation failed';
          } catch (err) {
            newTranslations[i] = 'Translation failed';
          }
        }
      }
      setTranslations(newTranslations);
    } catch (err) {
      setError('Failed to translate clauses');
    } finally {
      setIsTranslating(false);
    }
  };

  return (
    <div className="interface-container">
      <div className="interface-header">
        <h1 className="interface-title">⚖️ Legal Document Simplifier</h1>
        <p className="interface-subtitle">
          Upload rental agreements, contracts, or legal documents to get plain English explanations
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
            <div className="upload-icon">📁</div>
            <p className="upload-text">
              Drag & drop your legal document here or{' '}
              <span className="browse-button">browse</span>
            </p>
            <p className="upload-hint">Supports PDF, DOCX, TXT, and images</p>
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
              <span className="divider-text">OR paste text directly</span>
              <span className="divider-line"></span>
            </div>
            <textarea
              className="text-area"
              placeholder="Paste your legal document text here...

Example:
The Tenant shall pay a security deposit of Rs. 50,000 which shall be forfeited if the tenant vacates before completing 11 months of stay..."
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
                Processing...
              </>
            ) : (
              <>
                ✨ Simplify Document
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
          <p className="loading-text">Analyzing your document... This may take a moment.</p>
        </div>
      )}

      {results && (
        <div className="results-section">
          <div className="results-header">
            <h2 className="results-title">📋 Analysis Results</h2>
            <button className="clear-button" onClick={handleClear}>
              ← Process Another
            </button>
          </div>

          {/* Summary Card */}
          <div className="summary-card">
            <h3 className="summary-title">Document Summary</h3>
            <div className="summary-stats">
              <div className="stat-item">
                <div className="stat-value">{results.summary?.total_clauses || results.clauses?.length || 0}</div>
                <div className="stat-label">Clauses Found</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{results.summary?.high_risk_count || 0}</div>
                <div className="stat-label">High Risk</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{results.summary?.overall_risk || 'LOW'}</div>
                <div className="stat-label">Overall Risk</div>
              </div>
            </div>
          </div>

          {/* Global Column Headers with Translation Dropdown */}
          <div className="clause-column-headers">
            <div className="column-header-item">📄 Original</div>
            <div className="column-header-item">✨ Simplified</div>
            <div className="column-header-item translation-header">
              🌐 Translation
              <select 
                className="global-translation-select"
                value={selectedLang}
                onChange={(e) => handleTranslationChange(e.target.value as TranslationLang)}
                disabled={isTranslating}
              >
                <option value="none">Select Language</option>
                <option value="hi">हिंदी (Hindi)</option>
                <option value="ta">தமிழ் (Tamil)</option>
              </select>
              {isTranslating && <span className="translating-indicator">⏳</span>}
            </div>
            <div className="column-header-item">⚠️ Risk</div>
          </div>

          {/* Clause Rows */}
          {results.clauses?.map((clause: ClauseData, index: number) => (
            <ClauseRow 
              key={index} 
              clause={clause} 
              index={index + 1}
              translation={translations[index] || ''}
              isTranslating={isTranslating && selectedLang !== 'none'}
            />
          ))}

          {/* Law References Section */}
          {results.law_references && results.law_references.length > 0 && (
            <div className="law-references-section">
              <div className="clause-card">
                <div className="clause-header">
                  <span className="clause-number">📚 Laws Referenced in This Document</span>
                </div>
                <div className="clause-content">
                  <p style={{ marginBottom: '1rem', color: '#666' }}>
                    Click on any law below to learn what it means and how it applies to your document:
                  </p>
                  <div className="law-tags">
                    {results.law_references.map((law: LawReference, i: number) => (
                      <button 
                        key={i}
                        className="law-tag"
                        onClick={() => setSelectedLaw(law)}
                        title={`Click to learn about ${law.law_name}`}
                      >
                        <span className="law-tag-icon">⚖️</span>
                        <span className="law-tag-name">{law.law_name}</span>
                        {law.section && <span className="law-tag-section">§{law.section}</span>}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Law Popup */}
          {selectedLaw && (
            <LawPopup law={selectedLaw} onClose={() => setSelectedLaw(null)} />
          )}

          {/* Recommendations */}
          {results.summary?.recommendations && results.summary.recommendations.length > 0 && (
            <div className="clause-card">
              <div className="clause-header">
                <span className="clause-number">💡 Recommendations</span>
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

export default LegalInterface;
