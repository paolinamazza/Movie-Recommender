#!/bin/bash

# =============================================
# CineMatch AI - Web App Launcher
# =============================================

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║              🎬 CineMatch AI Web Interface 🎬              ║"
echo "║                                                            ║"
echo "║         Professional HTML Alternative to Streamlit        ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create a .env file with your OPENAI_API_KEY"
    exit 1
fi

# Check if venv exists
if [ ! -d venv ]; then
    echo "❌ Error: venv not found!"
    echo "Please run: python3 -m venv venv"
    exit 1
fi

# Activate venv
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install Flask dependencies if needed
echo "📦 Checking Flask dependencies..."
pip install flask flask-cors -q

# Launch Flask app
echo "🎟️ Starting Flask server..."
echo "🌐 Open your browser to: http://localhost:5001"
echo ""
echo "📝 Note: Using port 5001 (port 5000 is used by macOS AirPlay)"
echo "Press CTRL+C to stop"
echo ""

# Run the Flask app
python3 app/web_app.py
