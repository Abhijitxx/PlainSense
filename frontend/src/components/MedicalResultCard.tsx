import React, { useState } from 'react';
import { MedicalTestData } from '../services/api';

interface MedicalResultCardProps {
  data: MedicalTestData;
  index: number;
}

type LanguageOption = 'none' | 'hindi_plain' | 'hindi_colloquial' | 'tamil_plain' | 'tamil_colloquial';

const MedicalResultCard: React.FC<MedicalResultCardProps> = ({ data, index }) => {
  const [selectedLang, setSelectedLang] = useState<LanguageOption>('none');

  // Determine status from risk level or explicit status
  const getStatus = (): 'high' | 'low' | 'normal' => {
    const risk = data.risk?.level?.toLowerCase() || '';
    if (risk.includes('critical') || risk.includes('high')) return 'high';
    if (risk.includes('medium')) return 'low';
    return 'normal';
  };

  const status = getStatus();
  const statusText = status === 'high' ? 'ABNORMAL HIGH' : status === 'low' ? 'ABNORMAL LOW' : 'NORMAL';

  const getTranslation = (): string => {
    switch (selectedLang) {
      case 'hindi_plain':
        return data.hindi?.formal || 'Translation not available';
      case 'hindi_colloquial':
        return data.hindi?.colloquial || 'Translation not available';
      case 'tamil_plain':
        return data.tamil?.formal || 'Translation not available';
      case 'tamil_colloquial':
        return data.tamil?.colloquial || 'Translation not available';
      default:
        return '';
    }
  };

  const getTranslationClass = (): string => {
    if (selectedLang.includes('hindi')) return 'hindi';
    if (selectedLang.includes('tamil')) return 'tamil';
    return '';
  };

  // Medical explanations based on status
  const getHealthExplanation = () => {
    if (status === 'high') {
      return {
        title: '⚠️ What This Means',
        causes: [
          'Values above the normal range indicate potential health concerns',
          'Your body may be showing signs of imbalance or stress',
          'Further tests may be needed to determine the exact cause'
        ],
        effects: [
          'May cause symptoms like fatigue, discomfort, or other issues',
          'Could indicate an underlying condition that needs attention',
          'Early detection allows for better treatment outcomes'
        ],
        advice: 'Please consult with your doctor to discuss these results and determine the next steps.'
      };
    } else if (status === 'low') {
      return {
        title: '⚠️ What This Means',
        causes: [
          'Values below the normal range may indicate deficiencies',
          'Your body may not be producing or receiving enough of this substance',
          'Diet, lifestyle, or health conditions could be contributing factors'
        ],
        effects: [
          'May cause weakness, fatigue, or reduced immune function',
          'Could affect your overall energy levels and wellbeing',
          'Long-term deficiencies can lead to more serious health issues'
        ],
        advice: 'Discuss with your doctor about dietary changes, supplements, or further testing.'
      };
    }
    
    return {
      title: '✅ Good News',
      causes: [],
      effects: [],
      advice: 'Your values are within the healthy range. Continue maintaining a healthy lifestyle!'
    };
  };

  const healthInfo = getHealthExplanation();

  return (
    <div className="medical-result-card">
      <div className="test-header">
        <span className="test-name">🔬 Test Result {index}</span>
        <div className="test-value">
          <span className={`status-indicator ${status}`}></span>
          <span style={{ 
            color: status === 'high' ? '#ef4444' : status === 'low' ? '#3b82f6' : '#10b981',
            fontWeight: 600 
          }}>
            {statusText}
          </span>
        </div>
      </div>

      <div className="test-content">
        {/* Original Text */}
        <div className="content-block">
          <div className="content-label">Original Report Text</div>
          <div className="original-text">{data.original}</div>
        </div>

        {/* Simplified Explanation */}
        <div className="content-block">
          <div className="content-label">✅ Simple Explanation</div>
          <div className="simplified-text">
            {data.english?.plain || data.simplified || 'Understanding your result...'}
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

        {/* Health Explanation Section */}
        <div className="explanation-section">
          <div className="explanation-title">{healthInfo.title}</div>
          
          <div className={`explanation-content ${status === 'normal' ? 'normal' : ''}`}>
            {data.risk?.explanation || healthInfo.advice}
          </div>

          {status !== 'normal' && (
            <div className="causes-effects">
              {healthInfo.causes.length > 0 && (
                <>
                  <div className="content-label" style={{ marginTop: '1rem' }}>🔍 Possible Causes</div>
                  {healthInfo.causes.map((cause, i) => (
                    <div key={i} className="cause-effect-item">
                      <span className="ce-icon">•</span>
                      <span>{cause}</span>
                    </div>
                  ))}
                </>
              )}

              {healthInfo.effects.length > 0 && (
                <>
                  <div className="content-label" style={{ marginTop: '1rem' }}>⚡ Potential Effects</div>
                  {healthInfo.effects.map((effect, i) => (
                    <div key={i} className="cause-effect-item">
                      <span className="ce-icon">•</span>
                      <span>{effect}</span>
                    </div>
                  ))}
                </>
              )}

              <div style={{ 
                marginTop: '1rem', 
                padding: '0.75rem', 
                background: '#dbeafe', 
                borderRadius: '8px',
                fontSize: '0.9rem'
              }}>
                💡 <strong>Recommendation:</strong> {healthInfo.advice}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MedicalResultCard;
