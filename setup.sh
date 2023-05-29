#!/bin/bash

# Install Tesseract OCR and dependencies
apt-get install -y tesseract-ocr libtesseract-dev libleptonica-dev

# Install Python packages
pip install pytesseract==0.3.8 pillow==8.4.0

# Make sure the Tesseract OCR executable is in the PATH
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/

# Run the Streamlit app
streamlit run app.py
