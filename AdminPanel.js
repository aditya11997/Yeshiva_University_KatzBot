import React, { useState } from 'react';
import '../styles/AdminPanel.css'; // Custom styles

const AdminPanel = () => {
  const [urls, setUrls] = useState('');
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);

  const handleUpdate = async () => {
    try {
      const response = await fetch('http://127.0.0.1:5000/update_urls', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ urls }),
      });

      const data = await response.json();

      if (response.ok) {
        setMessage('URLs updated and data processed successfully!');
        setError(''); // Clear any previous error messages
      } else {
        setError(data.error || 'An error occurred while updating URLs');
        setMessage(''); // Clear any previous success messages
      }
    } catch (err) {
      setError('Network error. Please try again.');
      setMessage('');
      console.error('Error updating URLs:', err);
    }
  };

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleFileUpload = async () => {
    if (!selectedFile) {
      setError('Please select a PDF file to upload.');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://127.0.0.1:5000/upload_pdf', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok) {
        setMessage('PDF uploaded and data processed successfully!');
        setError(''); // Clear any previous error messages
      } else {
        setError(data.error || 'An error occurred while uploading the PDF');
        setMessage(''); // Clear any previous success messages
      }
    } catch (err) {
      setError('Network error. Please try again.');
      setMessage('');
      console.error('Error uploading PDF:', err);
    }
  };

  return (
    <div className="admin-container">
      <div className="admin-panel">
        <h2>Admin Panel</h2>
        {message && <p className="success-message">{message}</p>}
        {error && <p className="error-message">{error}</p>}
        <textarea
          value={urls}
          onChange={(e) => setUrls(e.target.value)}
          placeholder="Enter multiple URLs separated by commas"
          className="url-input"
        />
        <div className="button-container">
          <button className="update-button" onClick={handleUpdate}>
            Update
          </button>
        </div>

        <h3 className='upload-pdf-text'>Upload PDF</h3>
        <div className="upload-container">
          <label className="custom-file-upload">
            <input type="file" onChange={handleFileChange} accept=".pdf" />
            Choose File
          </label>
          <button className="upload-button" onClick={handleFileUpload}>
            Upload PDF
          </button>
        </div>
      </div>
    </div>
  );
};

export default AdminPanel;
