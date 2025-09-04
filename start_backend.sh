#!/bin/bash
echo "Starting BOM Platform Backend..."
cd backend
source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null || echo "No virtual environment found"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
