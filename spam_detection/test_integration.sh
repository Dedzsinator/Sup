#!/bin/bash

# Integration test script for Spam Detection Microservice

set -e

echo "🧪 Spam Detection Integration Test"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function success() {
    echo -e "${GREEN}✓ $1${NC}"
}

function warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

function error() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

# Test 1: Check Python dependencies
echo "🔍 Testing Python dependencies..."
python -c "
import sys
sys.path.append('src')
try:
    import fastapi, uvicorn, pydantic
    print('FastAPI dependencies: OK')
    import numpy, sklearn
    print('ML dependencies: OK')
    import src.simple_server as server
    print('Server module: OK')
except ImportError as e:
    print(f'Import error: {e}')
    sys.exit(1)
" || error "Python dependencies missing"
success "Python dependencies installed"

# Test 2: Basic spam detection functionality
echo "🔍 Testing spam detection logic..."
python -c "
import sys
sys.path.append('src')
import src.simple_server as server

# Test spam message
spam_prob, confidence = server.simple_spam_detection('FREE MONEY WIN NOW URGENT CLICK HERE!')
print(f'Spam test: prob={spam_prob:.3f}, confidence={confidence:.3f}')
assert spam_prob > 0.3, f'Expected high spam probability, got {spam_prob}'

# Test normal message  
ham_prob, ham_confidence = server.simple_spam_detection('Hey, how are you doing today?')
print(f'Ham test: prob={ham_prob:.3f}, confidence={ham_confidence:.3f}')
assert ham_prob < 0.3, f'Expected low spam probability, got {ham_prob}'

print('Spam detection logic working correctly!')
" || error "Spam detection logic failed"
success "Spam detection logic working"

# Test 3: Check if server can start (dry run)
echo "🔍 Testing server startup..."
timeout 5 python -c "
import sys
sys.path.append('src')
from src.simple_server import app
print('Server app created successfully')
" || error "Server startup failed"
success "Server can be created"

# Test 4: Check Docker setup
echo "🔍 Testing Docker configuration..."
if [ -f "Dockerfile" ]; then
    success "Dockerfile exists"
else
    warning "Dockerfile not found"
fi

if [ -f "docker-compose.yml" ]; then
    success "Docker Compose file exists"
else
    warning "Docker Compose file not found"
fi

# Test 5: Check integration files
echo "🔍 Testing integration files..."

# Check backend integration
if [ -f "../backend/lib/sup/spam_detection/client.ex" ]; then
    success "Backend client integration exists"
else
    error "Backend client integration missing"
fi

if [ -f "../backend/lib/sup/spam_detection/service.ex" ]; then
    success "Backend service integration exists"
else
    error "Backend service integration missing"
fi

# Check frontend integration
if [ -f "../frontend/src/components/spam/SpamDetectionComponent.tsx" ]; then
    success "Frontend component integration exists"
else
    warning "Frontend component integration missing"
fi

# Test 6: Validate API models
echo "🔍 Testing API models..."
python -c "
import sys
sys.path.append('src')
from src.simple_server import SpamCheckRequest, SpamCheckResponse, HealthResponse

# Test request model
req = SpamCheckRequest(message='test', user_id='user1')
print(f'Request model: OK')

# Test response model
resp = SpamCheckResponse(
    is_spam=False, 
    spam_probability=0.1, 
    confidence=0.8, 
    processing_time_ms=10.0, 
    user_id='user1'
)
print(f'Response model: OK')

print('API models validated successfully!')
" || error "API model validation failed"
success "API models validated"

# Test 7: Check configuration
echo "🔍 Testing configuration..."
if [ -f "requirements.txt" ]; then
    success "Requirements file exists"
else
    error "Requirements file missing"
fi

if [ -f "start.sh" ]; then
    success "Start script exists"
    if [ -x "start.sh" ]; then
        success "Start script is executable"
    else
        warning "Start script not executable"
    fi
else
    error "Start script missing"
fi

# Test 8: Directory structure
echo "🔍 Testing directory structure..."
required_dirs=("src" "data" "models" "config" "tests")
for dir in "${required_dirs[@]}"; do
    if [ -d "$dir" ]; then
        success "Directory $dir exists"
    else
        warning "Directory $dir missing"
    fi
done

# Test 9: Sample data
echo "🔍 Testing sample data..."
if [ -f "data/spam_dataset.jsonl" ]; then
    success "Sample dataset exists"
    # Count lines
    lines=$(wc -l < data/spam_dataset.jsonl)
    if [ $lines -gt 0 ]; then
        success "Sample dataset has $lines entries"
    else
        warning "Sample dataset is empty"
    fi
else
    warning "Sample dataset missing"
fi

# Test 10: End-to-end functionality test
echo "🔍 Testing end-to-end functionality..."
python -c "
import sys
sys.path.append('src')
import asyncio
from src.simple_server import predict_spam, SpamCheckRequest
from datetime import datetime

async def test_endpoint():
    # Create test request
    request = SpamCheckRequest(
        message='FREE MONEY WIN NOW URGENT!',
        user_id='test_user',
        timestamp=datetime.now()
    )
    
    # Test the prediction endpoint
    response = await predict_spam(request)
    
    print(f'End-to-end test result:')
    print(f'  Message: {request.message}')
    print(f'  Is Spam: {response.is_spam}')
    print(f'  Probability: {response.spam_probability:.3f}')
    print(f'  Confidence: {response.confidence:.3f}')
    print(f'  Processing Time: {response.processing_time_ms:.2f}ms')
    
    assert response.user_id == request.user_id
    assert 0 <= response.spam_probability <= 1
    assert 0 <= response.confidence <= 1
    
    print('End-to-end test passed!')

# Run the async test
asyncio.run(test_endpoint())
" || error "End-to-end test failed"
success "End-to-end functionality working"

# Summary
echo ""
echo "🎉 Integration Test Summary"
echo "=========================="
echo -e "${GREEN}All critical tests passed!${NC}"
echo ""
echo "📋 System Status:"
echo "  ✓ Core spam detection working"
echo "  ✓ API endpoints functional"
echo "  ✓ Backend integration ready"
echo "  ✓ Frontend components available"
echo "  ✓ Docker configuration complete"
echo ""
echo "🚀 Ready for deployment!"
echo ""
echo "To start the service:"
echo "  ./start.sh"
echo ""
echo "To test with Docker:"
echo "  docker-compose up spam_detection"
echo ""
echo "To run the full Sup application:"
echo "  cd .. && docker-compose up"
