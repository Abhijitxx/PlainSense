import React, { useState, useEffect } from 'react';
import { ClauseData } from '../services/api';

interface ClauseRowProps {
  clause: ClauseData;
  index: number;
  translation?: string;
  isTranslating?: boolean;
}

const ClauseRow: React.FC<ClauseRowProps> = ({ clause, index, translation = '', isTranslating = false }) => {
  const [displayedOriginal, setDisplayedOriginal] = useState('');
  const [displayedSimplified, setDisplayedSimplified] = useState('');
  const [displayedTranslation, setDisplayedTranslation] = useState('');
  
  const original = clause.original || '';
  const simplified = clause.english?.plain || clause.simplified || '';
  const riskLevel = clause.risk?.level || 'none';
  
  // Typing animation for original text
  useEffect(() => {
    setDisplayedOriginal('');
    let i = 0;
    const text = original;
    const timer = setInterval(() => {
      if (i < text.length) {
        setDisplayedOriginal(text.substring(0, i + 1));
        i++;
      } else {
        clearInterval(timer);
      }
    }, 5);
    return () => clearInterval(timer);
  }, [original]);
  
  // Typing animation for simplified text
  useEffect(() => {
    setDisplayedSimplified('');
    let i = 0;
    const text = simplified;
    const timer = setInterval(() => {
      if (i < text.length) {
        setDisplayedSimplified(text.substring(0, i + 1));
        i++;
      } else {
        clearInterval(timer);
      }
    }, 8);
    return () => clearInterval(timer);
  }, [simplified]);
  
  // Typing animation for translation (when received from parent)
  useEffect(() => {
    if (isTranslating) {
      setDisplayedTranslation('Translating...');
      return;
    }
    
    if (!translation) {
      setDisplayedTranslation('');
      return;
    }
    
    // Animate the translation text
    setDisplayedTranslation('');
    let i = 0;
    const timer = setInterval(() => {
      if (i < translation.length) {
        setDisplayedTranslation(translation.substring(0, i + 1));
        i++;
      } else {
        clearInterval(timer);
      }
    }, 8);
    return () => clearInterval(timer);
  }, [translation, isTranslating]);
  
  const getRiskBadgeClass = () => {
    const l = riskLevel.toLowerCase();
    if (l === 'critical') return 'risk-badge critical';
    if (l === 'high') return 'risk-badge high';
    if (l === 'medium') return 'risk-badge medium';
    if (l === 'low') return 'risk-badge low';
    return 'risk-badge none';
  };
  
  const getRiskEmoji = () => {
    const l = riskLevel.toLowerCase();
    if (l === 'critical') return '🔴';
    if (l === 'high') return '🟠';
    if (l === 'medium') return '🟡';
    if (l === 'low') return '🟢';
    return '✅';
  };

  return (
    <div className="clause-row">
      <div className="clause-row-header">
        <span className="clause-number">📜 Clause {index}</span>
      </div>
      
      <div className="clause-columns">
        {/* Column 1: Original */}
        <div className="clause-column original-column">
          <div className="column-header">Original Text</div>
          <div className="column-content typing-text">
            {displayedOriginal}
            {displayedOriginal.length < original.length && <span className="cursor">|</span>}
          </div>
        </div>
        
        {/* Column 2: Simplified */}
        <div className="clause-column simplified-column">
          <div className="column-header">✅ Plain English</div>
          <div className="column-content simplified-text typing-text">
            {displayedSimplified}
            {displayedSimplified.length < simplified.length && <span className="cursor">|</span>}
          </div>
        </div>
        
        {/* Column 3: Translation */}
        <div className="clause-column translation-column">
          <div className="column-header">🌐 Translation</div>
          <div className="column-content translation-text">
            {isTranslating ? (
              <span className="translating">
                <span className="spinner-small"></span>
                Translating...
              </span>
            ) : translation ? (
              <>
                {displayedTranslation}
                {displayedTranslation.length < translation.length && <span className="cursor">|</span>}
              </>
            ) : (
              <span className="placeholder-text">Select a language above to translate all clauses</span>
            )}
          </div>
        </div>
        
        {/* Column 4: Risk */}
        <div className="clause-column risk-column">
          <div className="column-header">⚠️ Risk Level</div>
          <div className="column-content risk-content">
            <div className={getRiskBadgeClass()}>
              {getRiskEmoji()} {riskLevel.toUpperCase()}
            </div>
            {clause.risk?.explanation && (
              <div className="risk-explanation">
                {clause.risk.explanation}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ClauseRow;
