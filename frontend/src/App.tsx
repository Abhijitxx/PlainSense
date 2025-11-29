import React, { useState } from 'react';
import './App.css';
import LegalInterface from './components/LegalInterface';
import MedicalInterface from './components/MedicalInterface';
import Header from './components/Header';

type TabType = 'legal' | 'medical';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('legal');

  return (
    <div className="app">
      <Header />
      
      <div className="tab-container">
        <button 
          className={`tab-button ${activeTab === 'legal' ? 'active' : ''}`}
          onClick={() => setActiveTab('legal')}
        >
          <span className="tab-icon">⚖️</span>
          <span className="tab-text">Legal Documents</span>
        </button>
        <button 
          className={`tab-button ${activeTab === 'medical' ? 'active' : ''}`}
          onClick={() => setActiveTab('medical')}
        >
          <span className="tab-icon">🏥</span>
          <span className="tab-text">Medical Reports</span>
        </button>
      </div>

      <main className="main-content">
        {activeTab === 'legal' ? <LegalInterface /> : <MedicalInterface />}
      </main>

      <footer className="footer">
        <p>PlainSense © 2025 - Making complex documents simple</p>
      </footer>
    </div>
  );
}

export default App;
