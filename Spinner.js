import React from 'react';
import '../styles/Spinner.css';  // Ensure you create a corresponding CSS file for styling

const Spinner = () => {
  return (
    <div className="spinner">
      <div className="double-bounce1"></div>
      <div className="double-bounce2"></div>
    </div>
  );
};

export default Spinner;
