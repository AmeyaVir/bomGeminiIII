#!/bin/bash

echo "🚀 Setting up BOM Platform Enhanced v3.0..."
echo "============================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Python is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

print_info "Python found: $PYTHON_CMD"

# Backend setup
print_info "Setting up backend..."
cd backend

# Create virtual environment if it doesn't exist
if [[ ! -d "venv" ]]; then
    print_info "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# Install dependencies
pip install -r requirements.txt
print_status "Backend dependencies installed"

# Create directories
mkdir -p uploads results
print_status "Created upload and results directories"

# Initialize database
$PYTHON_CMD -c "from models import init_db; init_db()"
print_status "Database initialized"

cd ..

# Frontend setup
print_info "Setting up frontend..."
cd frontend

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 16+"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ npm not found. Please install npm"
    exit 1
fi

print_status "Node.js and npm found"

# Install dependencies
npm install
print_status "Frontend dependencies installed"

cd ..

# Make scripts executable
chmod +x start_backend.sh start_frontend.sh

print_status "Setup complete!"
echo ""
echo "🚀 To start the platform:"
echo "  Backend:  ./start_backend.sh"
echo "  Frontend: ./start_frontend.sh"
echo ""
echo "📊 Access the application:"
echo "  Backend API: http://localhost:8000"
echo "  Frontend: http://localhost:3000"
