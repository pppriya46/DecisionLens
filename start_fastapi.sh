#!/bin/bash

# DecisionLens Phase 2 - Quick Start Script
# FastAPI Setup and Launch

set -e  # Exit on error

echo "=================================================="
echo "DecisionLens Phase 2: FastAPI Migration"
echo "Quick Start Script"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found"
    echo "Please run this script from the DecisionLens project root"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install/update dependencies
echo ""
echo "Installing FastAPI dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✓ Dependencies installed"

# Check for .env file
echo ""
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found"
    echo ""
    echo "Creating .env template..."
    cat > .env << 'EOF'
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=decisionlens_db
DB_USER=decisionlens
DB_PASSWORD=decisionlens123

# OpenAI API Key (required for embeddings and RAG)
OPENAI_API_KEY=your-openai-api-key-here

# API Configuration
PORT=5000
EOF
    echo "✓ Created .env template"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your OPENAI_API_KEY"
    echo ""
    read -p "Press Enter to continue..."
else
    echo "✓ .env file found"
fi

# Check if PostgreSQL is running
echo ""
echo "Checking database connection..."
if docker ps | grep -q "decisionlens_postgres_dev"; then
    echo "✓ PostgreSQL container is running"
else
    echo "⚠️  PostgreSQL container not running"
    echo ""
    read -p "Start Docker services? (y/n): " START_DOCKER
    if [ "$START_DOCKER" = "y" ]; then
        echo "Starting Docker services..."
        docker-compose -f docker-compose-dev.yml up -d postgres
        echo "Waiting for PostgreSQL to be ready..."
        sleep 5
        echo "✓ PostgreSQL started"
    else
        echo "⚠️  Database connection may fail. Start manually with:"
        echo "   docker-compose -f docker-compose-dev.yml up -d postgres"
    fi
fi

# Check if ML models exist
echo ""
echo "Checking ML models..."
if [ -f "ml/models/severity_rf_v1.pkl" ]; then
    echo "✓ Severity model found"
else
    echo "⚠️  Severity model not found"
    echo "   Run: python ml/severity_model.py (to train the model)"
fi

# Display startup options
echo ""
echo "=================================================="
echo "Setup Complete! Choose how to start the API:"
echo "=================================================="
echo ""
echo "Option 1: Local Development (Recommended for debugging)"
echo "  uvicorn api.main:app --reload --host 0.0.0.0 --port 5000"
echo ""
echo "Option 2: Docker Compose (Full stack with auto-reload)"
echo "  docker-compose -f docker-compose-dev.yml up --build"
echo ""
echo "Option 3: Direct Python execution"
echo "  python -m api.main"
echo ""
echo "=================================================="
echo ""

read -p "Start with Option 1 (local uvicorn)? (y/n): " START_LOCAL

if [ "$START_LOCAL" = "y" ]; then
    echo ""
    echo "Starting FastAPI server..."
    echo "=================================================="
    echo "Access points:"
    echo "  - API Root:        http://localhost:5000/"
    echo "  - Swagger Docs:    http://localhost:5000/docs"
    echo "  - ReDoc:           http://localhost:5000/redoc"
    echo "  - Health Check:    http://localhost:5000/health"
    echo "=================================================="
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo ""
    
    # Start uvicorn with reload
    uvicorn api.main:app --reload --host 0.0.0.0 --port 5000
else
    echo ""
    echo "To start manually, run:"
    echo "  source venv/bin/activate"
    echo "  uvicorn api.main:app --reload --host 0.0.0.0 --port 5000"
    echo ""
    echo "Or with Docker:"
    echo "  docker-compose -f docker-compose-dev.yml up --build"
fi
