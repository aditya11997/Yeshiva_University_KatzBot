import React, { useState, useEffect, useRef } from 'react';
import '../styles/Sidebar.css';  // Ensure you have the styles referenced here
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';  // Import FontAwesome component
import { faChevronLeft, faChevronRight } from '@fortawesome/free-solid-svg-icons';

const Sidebar = ({ selectedLanguage, setSelectedLanguage }) => {
  const [isOpen, setIsOpen] = useState(false);  // Sidebar open/close state
  const sidebarRef = useRef(null);

  // Function to toggle the sidebar
  const toggleSidebar = () => {
    setIsOpen(!isOpen);
  };

  // Function to handle language change
  const handleLanguageChange = (e) => {
    setSelectedLanguage(e.target.value);  // Update language in parent component
  };

  // Close sidebar if clicked outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (sidebarRef.current && !sidebarRef.current.contains(event.target)) {
        setIsOpen(false);  // Close sidebar if the click is outside
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);  // Cleanup event listener
    };
  }, [sidebarRef]);

  return (
    <>
      <div 
        className={`sidebar ${isOpen ? 'open' : 'collapsed'}`} 
        ref={sidebarRef}
      >
        <div className="sidebar-header">
          {/* Show the arrow when hovered */}
          <div className="collapse-icon" onClick={toggleSidebar}>
            {isOpen ? (
                <FontAwesomeIcon icon={faChevronLeft} />
              ) : (
                <FontAwesomeIcon icon={faChevronRight} />
              )}
          </div>
        </div>
        {/* Sidebar content */}
        {isOpen && (
          <div className="sidebar-content">
            <div className='label'>Select Language</div>
            <select 
              className='sidebar-language-selection' 
              value={selectedLanguage} // Set selected language
              onChange={handleLanguageChange}
            >
              <option value="en-US">English</option>
              <option value="zh-CN">中文</option>
            </select>
          </div>
        )}
      </div>

      {/* Collapse icon outside of the sidebar to ensure it's always visible */}
      {!isOpen && (
        <div className="collapse-icon" onClick={toggleSidebar}>
          <FontAwesomeIcon icon={faChevronRight} />
        </div>
      )}
    </>
  );
};

export default Sidebar;
