"""
Custom CSS Styling

This module provides custom styling for the Plant Disease Detector frontend
with responsive design, animations, and professional appearance.
"""

import streamlit as st
from typing import Dict, Any


def setup_custom_css():
    """Apply custom CSS styling to the Streamlit app"""
    css_style = """
    /* Main container and layout */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
        background-size: cover;
        min-height: 100vh;
    }

    .main-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Header styling */
    .main-header {
        text-align: center;
        margin-bottom: 30px;
        padding: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }

    .header-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 15px;
    }

    .header-description {
        font-size: 1.2em;
        opacity: 0.9;
        max-width: 600px;
        line-height: 1.5;
    }

    .header-stats {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-top: 20px;
    }

    .stat-item {
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 5px;
    }

    .stat-number {
        font-size: 2.2em;
        font-weight: bold;
        background: linear-gradient(45deg, #4CAF50, #8BC34A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
    }

    .stat-label {
        font-size: 0.9em;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Uploader component styling */
    .uploader-header {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        text-align: center;
    }

    .uploader-title {
        font-size: 1.4em;
        margin-bottom: 10px;
        color: #1f2937;
        font-weight: 600;
    }

    .uploader-status {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 10px;
        padding: 8px 15px;
        border-radius: 20px;
        font-size: 0.9em;
    }

    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background-color: #4CAF50;
        animation: pulse 2s infinite;
    }

    .status-indicator.loading {
        background-color: #FF9800;
        animation: pulse 1s infinite;
    }

    .status-indicator.error {
        background-color: #F44336;
        animation: pulse 0.5s infinite;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }

    /* Quality metrics styling */
    .quality-score {
        text-align: center;
        margin: 20px 0;
        padding: 15px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .score-circle {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: conic-gradient(
            from 0deg,
            #4CAF50 0deg,
            #FF9800 33deg,
            #F44336 66deg,
            #f0f0f0 100deg
        );
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 15px;
        position: relative;
    }

    .score-circle::before {
        content: '';
        position: absolute;
        width: 60px;
        height: 60px;
        background: white;
        border-radius: 50%;
        top: 10px;
        left: 10px;
    }

    .score-label {
        font-weight: bold;
        font-size: 0.9em;
        text-transform: uppercase;
        color: #666;
    }

    .file-info {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 10px;
        margin-top: 15px;
    }

    .info-item {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 6px;
        border-left: 3px solid #667eea;
    }

    .info-label {
        font-weight: 600;
        color: #666;
        font-size: 0.8em;
    }

    .info-value {
        font-weight: 500;
        color: #1f2937;
        margin-top: 2px;
    }

    /* Tips section */
    .tip-item {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        margin-bottom: 8px;
        padding: 10px 15px;
        border-radius: 0 6px 6px 0;
        transition: all 0.3s ease;
    }

    .tip-item:hover {
        background: #bbdefb;
        transform: translateX(5px);
    }

    .tip-icon {
        margin-right: 8px;
        font-size: 1.1em;
    }

    /* Validation results */
    .validation-issues {
        background: #fff3e0;
        border-left: 4px solid #f44336;
        margin-top: 20px;
        padding: 15px;
        border-radius: 8px;
    }

    .recommendation {
        background: white;
        border-left: 4px solid #ff9800;
        margin-bottom: 10px;
        padding: 12px 15px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .recommendation-issue {
        font-weight: 600;
        color: #f44336;
        margin-bottom: 5px;
    }

    .recommendation-fix {
        color: #666;
        margin-top: 3px;
    }

    /* Results page styling */
    .results-header {
        text-align: center;
        margin-bottom: 30px;
        padding: 25px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    .results-summary {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }

    .summary-item {
        background: white;
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .summary-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }

    .summary-icon {
        font-size: 2em;
        margin-bottom: 10px;
        display: block;
    }

    .summary-content h3 {
        margin: 0;
        color: #1f2937;
        font-weight: 600;
    }

    .summary-content p {
        margin: 5px 0;
        color: #666;
        font-size: 0.9em;
    }

    .confidence-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
        margin-top: 8px;
    }

    .confidence-badge.good {
        background: #4CAF50;
        color: white;
    }

    .confidence-badge.medium {
        background: #FF9800;
        color: white;
    }

    .confidence-badge.low {
        background: #F44336;
        color: white;
    }

    /* Metric cards */
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
    }

    .metric-label {
        font-size: 0.9em;
        color: #666;
        font-weight: 500;
        margin-bottom: 5px;
    }

    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
        margin: 5px 0;
    }

    .metric-description {
        font-size: 0.8em;
        color: #999;
        margin-top: 5px;
    }

    /* Alternative identifications */
    .alternative-item {
        background: #f5f5f5;
        padding: 10px 15px;
        border-radius: 6px;
        margin-bottom: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border: 1px solid #e0e0e0;
    }

    .alternative-name {
        font-weight: 500;
        color: #1f2937;
    }

    .alternative-confidence {
        background: #667eea;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
    }

    /* Model performance */
    .model-performance {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }

    .model-row {
        display: grid;
        grid-template-columns: 2fr 1fr 1fr 1fr;
        gap: 15px;
        padding: 10px 0;
        border-bottom: 1px solid #f0f0f0;
    }

    .model-row:last-child {
        border-bottom: none;
    }

    /* Image comparison */
    .image-comparison {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
        margin: 20px 0;
    }

    .comparison-column {
        text-align: center;
    }

    .comparison-header {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 10px;
        padding: 10px;
        background: #f8f9fa;
        border-radius: 6px;
    }

    .comparison-image {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .comparison-image img {
        width: 100%;
        height: auto;
        display: block;
    }

    .comparison-info {
        text-align: center;
        margin-top: 10px;
        font-size: 0.9em;
        color: #666;
    }

    /* Treatment card styling */
    .treatment-header {
        text-align: center;
        margin-bottom: 30px;
        padding: 25px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .treatment-title {
        font-size: 1.6em;
        margin-bottom: 10px;
        font-weight: 600;
    }

    .treatment-priority {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 15px;
    }

    .priority-badge {
        padding: 6px 15px;
        border-radius: 20px;
        font-size: 0.9em;
        font-weight: 600;
    }

    .priority-badge.organic {
        background: #4CAF50;
        color: white;
    }

    .priority-badge.chemical {
        background: #F44336;
        color: white;
    }

    /* Calculator section */
    .calculator-section {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .section-title {
        font-size: 1.3em;
        color: #1f2937;
        margin-bottom: 15px;
        font-weight: 600;
    }

    /* Organic treatments */
    .treatment-method {
        background: white;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 15px;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
    }

    .treatment-method.chemical {
        border-left-color: #F44336;
    }

    .method-header {
        display: grid;
        grid-template-columns: 2fr 1fr;
        gap: 10px;
        margin-bottom: 15px;
        align-items: center;
    }

    .method-name {
        font-size: 1.1em;
        font-weight: 600;
        color: #1f2937;
    }

    .method-number {
        background: #667eea;
        color: white;
        width: 25px;
        height: 25px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 0.9em;
    }

    .method-content {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
    }

    .method-section {
        background: #f8f9fa;
        padding: 10px;
        border-radius: 6px;
    }

    .method-section h4 {
        margin: 0 0 5px;
        color: #1f2937;
        font-size: 0.9em;
        font-weight: 600;
    }

    .method-section p {
        margin: 0;
        color: #666;
        font-size: 0.85em;
    }

    /* Cultural practices */
    .practice-card {
        background: #e8f5e8;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 3px solid #4CAF50;
    }

    .practice-icon {
        font-size: 1.2em;
        margin-right: 8px;
        vertical-align: middle;
    }

    .practice-text {
        display: inline-block;
        vertical-align: middle;
        color: #2e7d32;
    }

    /* Prevention strategies */
    .prevention-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 15px;
    }

    .prevention-item {
        background: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
    }

    .prevention-icon {
        font-size: 1.5em;
        margin-bottom: 10px;
        display: block;
    }

    .prevention-content h4 {
        margin: 0;
        color: #1976d2;
        font-size: 1em;
    }

    .prevention-content p,
    .prevention-content ul {
        margin: 8px 0 0;
        color: #666;
        font-size: 0.9em;
    }

    .prevention-content ul {
        padding-left: 20px;
    }

    .prevention-content li {
        margin-bottom: 5px;
    }

    /* Professional consultation */
    .consultation-condition {
        display: flex;
        align-items: center;
        background: #f3e5f5;
        padding: 10px 15px;
        border-radius: 20px;
        margin-bottom: 10px;
        gap: 8px;
    }

    .condition-severity {
        font-size: 1.2em;
        margin-right: 5px;
    }

    .condition-text {
        font-weight: 500;
        color: #666;
    }

    .contact-resources {
        background: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        margin-top: 20px;
    }

    .contact-resources h4 {
        color: #1f2937;
        margin-bottom: 10px;
    }

    /* Confidence bars */
    .confidence-bar-container {
        margin: 15px 0;
        background: #f0f0f0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.1);
    }

    .confidence-label {
        padding: 8px 15px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: 600;
        text-align: center;
    }

    .confidence-bar {
        height: 8px;
        background: #e0e0e0;
        position: relative;
    }

    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        border-radius: 6px;
        transition: width 0.5s ease-in-out;
    }

    .confidence-fill.medium {
        background: linear-gradient(90deg, #FF9800, #F57C00);
    }

    .confidence-fill.low {
        background: linear-gradient(90deg, #F44336, #D32F2F);
    }

    .confidence-value {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        color: white;
        font-weight: bold;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
    }

    .confidence-status {
        text-align: center;
        margin-top: 8px;
        font-weight: 600;
        font-size: 0.9em;
    }

    /* Timeline for treatment plan */
    .treatment-timeline {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }

    .timeline-item {
        display: flex;
        align-items: flex-start;
        margin-bottom: 20px;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .timeline-item.immediate {
        background: #e8f5e8;
        border-left: 4px solid #4CAF50;
    }

    .timeline-item.short-term {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
    }

    .timeline-item.ongoing {
        background: #f3e5f5;
        border-left: 4px solid #ff9800;
    }

    .timeline-marker {
        font-size: 1.5em;
        margin-right: 15px;
        width: 30px;
        text-align: center;
    }

    .timeline-content {
        flex: 1;
    }

    .timeline-content h4 {
        margin: 0 0 5px;
        color: #1f2937;
    }

    .timeline-content p {
        margin: 0;
        color: #666;
        line-height: 1.4;
    }

    /* Cost analysis */
    .cost-summary {
        background: white;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .cost-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #f0f0f0;
    }

    .cost-item:last-child {
        border-bottom: none;
    }

    .cost-label {
        font-weight: 600;
        color: #1f2937;
    }

    .cost-value {
        font-weight: bold;
        color: #4CAF50;
    }

    .probability-item {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }

    .probability-bar {
        width: 100px;
        height: 8px;
        background: #e0e0e0;
        border-radius: 4px;
        overflow: hidden;
        margin-right: 10px;
    }

    .probability-bar.high {
        background: #4CAF50;
        width: 90px;
    }

    .probability-bar.medium {
        background: #FF9800;
        width: 60px;
    }

    .probability-bar.low {
        background: #F44336;
        width: 30px;
    }

    /* Feature list */
    .feature-list {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }

    .feature-item {
        background: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .feature-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
    }

    .feature-icon {
        font-size: 2.5em;
        margin-bottom: 15px;
        display: block;
    }

    .feature-text {
        color: #666;
        font-size: 0.9em;
        line-height: 1.4;
    }

    .feature-text strong {
        color: #1f2937;
        font-size: 1.1em;
    }

    /* Success stories */
    .story-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
        transition: all 0.3s ease;
    }

    .story-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .story-crop {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 5px;
        font-size: 1.1em;
    }

    .story-issue {
        color: #666;
        font-size: 0.9em;
        margin-bottom: 8px;
    }

    .story-result {
        color: #4CAF50;
        font-weight: 600;
    }

    /* Error display */
    .error-container {
        background: #fff3e0;
        border-left: 4px solid #f44336;
        padding: 15px 20px;
        border-radius: 8px;
        margin: 20px 0;
    }

    .error-message {
        color: #d32f2f;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .error-actions {
        margin-top: 15px;
    }

    /* Mobile optimizations */
    .mobile-notice {
        background: #fff3cd;
        border-left: 4px solid #ff9800;
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        text-align: center;
    }

    .mobile-notice h4 {
        color: #f57c00;
        margin-bottom: 10px;
    }

    /* Loading states */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(102, 126, 234, 0.3);
        border-radius: 50%;
        border-top-color: #667eea;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Animations */
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .slide-up {
        animation: slideUp 0.3s ease-out;
    }

    @keyframes slideUp {
        from { transform: translateY(10px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    /* Footer */
    .app-footer {
        background: white;
        padding: 30px;
        border-radius: 12px 12px 0 0;
        margin-top: 40px;
        box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.1);
        text-align: center;
    }

    .footer-content {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-bottom: 20px;
    }

    .footer-section h4 {
        color: #1f2937;
        margin-bottom: 10px;
    }

    .footer-section ul {
        list-style: none;
        padding: 0;
    }

    .footer-section li {
        margin-bottom: 5px;
    }

    .footer-section a {
        color: #667eea;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.3s ease;
    }

    .footer-section a:hover {
        color: #764ba2;
        text-decoration: underline;
    }

    .footer-bottom {
        text-align: center;
        margin-top: 20px;
        padding-top: 20px;
        border-top: 1px solid #e0e0e0;
        color: #999;
        font-size: 0.9em;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-container {
            padding: 15px;
            margin: 10px;
        }

        .header-stats {
            flex-direction: column;
            gap: 15px;
        }

        .results-summary {
            grid-template-columns: 1fr;
            gap: 15px;
        }

        .feature-list {
            grid-template-columns: 1fr;
        }

        .image-comparison {
            grid-template-columns: 1fr;
            gap: 15px;
        }

        .treatment-priority {
            flex-direction: column;
            gap: 10px;
        }

        .prevention-grid {
            grid-template-columns: 1fr;
        }

        .footer-content {
            grid-template-columns: 1fr;
            gap: 15px;
        }
    }

    @media (max-width: 480px) {
        .main-container {
            padding: 10px;
            margin: 5px;
        }

        .header-description {
            font-size: 1em;
        }

        .stat-number {
            font-size: 1.8em;
        }

        .feature-icon {
            font-size: 2em;
        }
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .main-container {
            background: rgba(18, 18, 18, 0.95);
            border-color: rgba(255, 255, 255, 0.1);
        }

        .uploader-header,
        .metric-card,
        .treatment-method,
        .story-card,
        .app-footer {
            background: #1f2937;
            color: white;
        }

        .info-item {
            background: rgba(255, 255, 255, 0.1);
            border-left-color: #667eea;
        }

        .info-label,
        .method-section h4,
        .prevention-content h4 {
            color: rgba(255, 255, 255, 0.9);
        }

        .info-value,
        .method-section p,
        .prevention-content p,
        .method-section li {
            color: rgba(255, 255, 255, 0.8);
        }
    }

    /* High contrast mode */
    @media (prefers-contrast: high) {
        .main-container {
            border: 2px solid #000;
        }

        .feature-item,
        .treatment-method {
            border: 1px solid #000;
        }
    }

    /* Reduced motion */
    @media (prefers-reduced-motion: reduce) {
        .loading-spinner,
        .status-indicator,
        .fade-in,
        .slide-up {
            animation: none;
        }
    }

    /* Print styles */
    @media print {
        .main-container {
            box-shadow: none;
            border: none;
        }

        .app-footer {
            box-shadow: none;
        }

        .feature-item {
            break-inside: avoid;
        }

        .treatment-method {
            break-inside: avoid;
        }
    }
    """

    st.markdown(css_style, unsafe_allow_html=True)


def create_confidence_bar(confidence: float, label: str) -> str:
    """Create HTML for confidence indicator bar"""
    if confidence >= 0.8:
        color = "#4CAF50"  # Green
        width_percent = confidence * 100
        status = "Excellent"
        fill_class = ""
    elif confidence >= 0.6:
        color = "#FF9800"  # Orange
        width_percent = confidence * 100
        status = "Good"
        fill_class = "medium"
    else:
        color = "#F44336"  # Red
        width_percent = confidence * 100
        status = "Needs Improvement"
        fill_class = "low"

    return f"""
    <div class="confidence-bar-container">
        <div class="confidence-label">{label}</div>
        <div class="confidence-bar">
            <div class="confidence-fill {fill_class}" style="width: {width_percent}%; background: {color};">
                <div class="confidence-value">{confidence:.0%}</div>
            </div>
        </div>
        <div class="confidence-status" style="color: {color};">{status} Confidence</div>
    </div>
    """


def create_comparison_layout(uploaded_image: bytes, disease_name: str) -> str:
    """Create HTML for side-by-side image comparison"""
    import base64

    # Convert uploaded image to base64
    uploaded_b64 = base64.b64encode(uploaded_image).decode()

    # Reference image paths (these would be actual paths in production)
    reference_images = [
        f"database/reference_images/{disease_name.lower().replace(' ', '_')}_1.jpg",
        f"database/reference_images/{disease_name.lower().replace(' ', '_')}_2.jpg"
    ]

    return f"""
    <div class="image-comparison">
        <div class="comparison-column">
            <div class="comparison-header">
                <h4>Your Image</h4>
            </div>
            <div class="comparison-image">
                <img src="data:image/jpeg;base64,{uploaded_b64}" alt="Uploaded leaf image">
            </div>
        </div>
        <div class="comparison-column">
            <div class="comparison-header">
                <h4>Reference Images</h4>
                <p>Typical {disease_name} symptoms</p>
            </div>
            <div class="reference-images">
                {''.join([f'''
                <div class="reference-image">
                    <img src="{img}" alt="Reference image {i+1}" onerror="this.style.display='none';">
                    <div class="reference-label">Example {i+1}</div>
                </div>
                ''' for i, img in enumerate(reference_images)])}
            </div>
        </div>
    </div>
    """


def create_metric_box(title: str, value: str, icon: str, color: str = "#667eea") -> str:
    """Create HTML for metric display box"""
    return f"""
    <div class="metric-card" style="border-color: {color};">
        <div class="metric-label">{icon} {title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


def create_alert_box(message: str, alert_type: str = "info") -> str:
    """Create HTML for alert/message box"""
    icons = {
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "info": "ℹ️"
    }

    colors = {
        "success": "#4CAF50",
        "warning": "#FF9800",
        "error": "#F44336",
        "info": "#2196F3"
    }

    icon = icons.get(alert_type, "ℹ️")
    color = colors.get(alert_type, "#2196F3")

    return f"""
    <div class="alert-box" style="background: {color}; border-left: 4px solid {color};">
        <div class="alert-content">
            <span class="alert-icon">{icon}</span>
            <span class="alert-message">{message}</span>
        </div>
    </div>
    """

    # Additional CSS for the custom components
    custom_css = """
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin: 15px 0;
        color: white;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    .alert-content {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .alert-icon {
        font-size: 1.2em;
    }

    .alert-message {
        font-weight: 500;
        line-height: 1.4;
    }
    """

    st.markdown(custom_css, unsafe_allow_html=True)


def setup_mobile_optimizations():
    """Add mobile-specific optimizations"""
    if st.sidebar.checkbox("Enable Mobile Mode", value=False):
        st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }

            @media (max-width: 768px) {
                .main-container {
                    padding: 10px;
                    margin: 5px;
                }

                .stMarkdown {
                    font-size: 0.9em;
                }

                .stButton {
                    width: 100% !important;
                    margin: 5px 0;
                }

                .stFileUploader {
                    padding: 10px;
                }
            }
        </style>
        """, unsafe_allow_html=True)

        st.session_state.mobile_optimized = True
    else:
        st.session_state.mobile_optimized = False


def setup_theme_preference(theme: str):
    """Apply theme preference"""
    if theme == "dark":
        st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(135deg, #1a237e 0%, #21a366 100%);
            }

            .main-container {
                background: rgba(26, 35, 126, 0.95);
                border-color: rgba(255, 255, 255, 0.1);
            }
        </style>
        """, unsafe_allow_html=True)
    elif theme == "light":
        st.markdown("""
        <style>
            .stApp {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            }

            .main-container {
                background: rgba(255, 255, 255, 0.95);
                border-color: rgba(0, 0, 0, 0.1);
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        # Auto/default theme
        st.markdown("""
        <style>
            @media (prefers-color-scheme: dark) {
                .stApp {
                    background: linear-gradient(135deg, #1a237e 0%, #21a366 100%);
                }
            }

            @media (prefers-color-scheme: light) {
                .stApp {
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                }
            }
        </style>
        """, unsafe_allow_html=True)


def setup_animations(animations_enabled: bool = True):
    """Enable or disable animations"""
    if animations_enabled:
        st.markdown("""
        <style>
            .feature-item,
            .story-card,
            .treatment-method {
                transition: all 0.3s ease;
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <style>
            .feature-item,
            .story-card,
            .treatment-method,
            .loading-spinner,
            .status-indicator {
                transition: none !important;
                animation: none !important;
            }
        </style>
        """, unsafe_allow_html=True)


def setup_accessibility_options(high_contrast: bool = False, reduced_motion: bool = False):
    """Setup accessibility options"""
    css_rules = []

    if high_contrast:
        css_rules.extend([
            ".main-container { border: 2px solid #000 !important; }",
            ".stButton { border: 2px solid #000 !important; }",
            ".feature-item { border: 1px solid #000 !important; }"
        ])

    if reduced_motion:
        css_rules.extend([
            ".loading-spinner { animation: none !important; }",
            ".status-indicator { animation: none !important; }",
            ".feature-item { transition: none !important; }",
            ".story-card { transition: none !important; }"
        ])

    if css_rules:
        st.markdown(f"""
        <style>
            {'; '.join(css_rules)}
        </style>
        """, unsafe_allow_html=True)