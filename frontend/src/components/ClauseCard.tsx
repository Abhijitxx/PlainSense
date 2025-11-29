import React, { useState } from 'react';
import { ClauseData } from '../services/api';

interface ClauseCardProps {
  clause: ClauseData;
  index: number;
}

type LanguageOption = 'none' | 'hindi_plain' | 'hindi_colloquial' | 'tamil_plain' | 'tamil_colloquial';

const ClauseCard: React.FC<ClauseCardProps> = ({ clause, index }) => {
  const [selectedLang, setSelectedLang] = useState<LanguageOption>('none');

  const getRiskClass = (level: string) => {
    const l = level?.toLowerCase() || 'none';
    if (l.includes('critical')) return 'critical';
    if (l.includes('high')) return 'high';
    if (l.includes('medium')) return 'medium';
    if (l.includes('low')) return 'low';
    return 'none';
  };

  const getRiskSectionClass = (level: string) => {
    const l = level?.toLowerCase() || 'none';
    if (l.includes('low') || l.includes('none')) return 'low';
    if (l.includes('medium')) return 'medium';
    return '';
  };

  const riskLevel = clause.risk?.level || 'NONE';
  const riskClass = getRiskClass(riskLevel);

  const getTranslation = (): string => {
    switch (selectedLang) {
      case 'hindi_plain':
        return clause.hindi?.formal || 'Translation not available';
      case 'hindi_colloquial':
        return clause.hindi?.colloquial || 'Translation not available';
      case 'tamil_plain':
        return clause.tamil?.formal || 'Translation not available';
      case 'tamil_colloquial':
        return clause.tamil?.colloquial || 'Translation not available';
      default:
        return '';
    }
  };

  const getTranslationClass = (): string => {
    if (selectedLang.includes('hindi')) return 'hindi';
    if (selectedLang.includes('tamil')) return 'tamil';
    return '';
  };

  return (
    <div className="clause-card">
      <div className="clause-header">
        <span className="clause-number">📜 Clause {index}</span>
        <span className={`risk-badge ${riskClass}`}>
          {riskLevel} Risk
        </span>
      </div>
      
      <div className="clause-content">
        {/* Original Text */}
        <div className="content-block">
          <div className="content-label">Original Text</div>
          <div className="original-text">{clause.original}</div>
        </div>

        {/* Simplified Text */}
        <div className="content-block">
          <div className="content-label">✅ Plain English</div>
          <div className="simplified-text">
            {clause.english?.plain || clause.simplified || 'Simplification not available'}
          </div>
        </div>

        {/* Language Selection */}
        <div className="language-section">
          <div className="content-label">🌐 View in Other Languages</div>
          <div className="language-tabs">
            <button 
              className={`lang-tab ${selectedLang === 'none' ? 'active' : ''}`}
              onClick={() => setSelectedLang('none')}
            >
              English Only
            </button>
            <button 
              className={`lang-tab ${selectedLang === 'hindi_plain' ? 'active' : ''}`}
              onClick={() => setSelectedLang('hindi_plain')}
            >
              🇮🇳 Hindi (Plain)
            </button>
            <button 
              className={`lang-tab ${selectedLang === 'hindi_colloquial' ? 'active' : ''}`}
              onClick={() => setSelectedLang('hindi_colloquial')}
            >
              🇮🇳 Hindi (Colloquial)
            </button>
            <button 
              className={`lang-tab ${selectedLang === 'tamil_plain' ? 'active' : ''}`}
              onClick={() => setSelectedLang('tamil_plain')}
            >
              தமிழ் (Plain)
            </button>
            <button 
              className={`lang-tab ${selectedLang === 'tamil_colloquial' ? 'active' : ''}`}
              onClick={() => setSelectedLang('tamil_colloquial')}
            >
              தமிழ் (Colloquial)
            </button>
          </div>

          {selectedLang !== 'none' && (
            <div className={`translation-box ${getTranslationClass()}`}>
              {getTranslation()}
            </div>
          )}
        </div>

        {/* Risk Explanation */}
        {clause.risk?.explanation && (
          <div className={`risk-section ${getRiskSectionClass(riskLevel)}`}>
            <div className="risk-label">⚠️ Risk Analysis</div>
            <div className="content-text">{clause.risk.explanation}</div>
          </div>
        )}

        {/* Key Terms */}
        {clause.key_terms && clause.key_terms.length > 0 && (
          <div className="content-block" style={{ marginTop: '1rem' }}>
            <div className="content-label">🔑 Key Terms</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
              {clause.key_terms.map((item, i: number) => (
                <span 
                  key={i} 
                  title={item.simplified}
                  style={{
                    padding: '0.25rem 0.75rem',
                    background: '#e2e8f0',
                    borderRadius: '20px',
                    fontSize: '0.875rem',
                    cursor: 'help'
                  }}
                >
                  {item.term}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ClauseCard;
