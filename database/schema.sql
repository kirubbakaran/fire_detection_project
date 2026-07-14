-- Fire Detection System - Database Schema
-- Run this once in MySQL to create the database and table.

CREATE DATABASE IF NOT EXISTS fire_detection_db;
USE fire_detection_db;

-- Stores every visit: who used the site and when
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    visit_time DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Optional: stores every detection result for history/analytics
CREATE TABLE IF NOT EXISTS detection_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL,
    detection_mode VARCHAR(20) NOT NULL,   -- 'camera' or 'photo'
    result VARCHAR(20) NOT NULL,           -- 'FIRE' or 'NO_FIRE'
    confidence FLOAT NOT NULL,
    detected_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
