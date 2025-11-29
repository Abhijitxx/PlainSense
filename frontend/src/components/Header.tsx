import React from 'react';

const Header: React.FC = () => {
  return (
    <header className="header">
      <div className="logo">
        <span className="logo-icon">📄</span>
        <span className="logo-text">PlainSense</span>
      </div>
      <span className="tagline">Simplify Legal & Medical Documents</span>
    </header>
  );
};

export default Header;
