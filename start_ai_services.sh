#!/bin/bash
# Start all AI services for the Sup application
#
# This script starts:
# 1. Spam Detection Service (port 8082)
# 2. Autocomplete Service (port 8000)

echo "🚀 Starting AI Services for Sup Application..."

# Function to check if a service is running
check_service() {
    local port=$1
    local service_name=$2
    
    if curl -s http://localhost:$port/health > /dev/null 2>&1; then
        echo "✅ $service_name is already running on port $port"
        return 0
    else
        echo "❌ $service_name is not running on port $port"
        return 1
    fi
}

# Function to start a service in the background
start_service() {
    local dir=$1
    local command=$2
    local service_name=$3
    local port=$4
    
    echo "🔄 Starting $service_name..."
    cd "$dir"
    
    # Start the service in the background
    nohup $command > "${service_name,,}_service.log" 2>&1 &
    local pid=$!
    
    echo "   Started $service_name with PID: $pid"
    
    # Wait a moment for the service to start
    sleep 3
    
    # Check if the service is responding
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:$port/health > /dev/null 2>&1; then
            echo "   ✅ $service_name is responding on port $port"
            return 0
        fi
        
        echo "   ⏳ Waiting for $service_name to start (attempt $attempt/$max_attempts)..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo "   ❌ $service_name failed to start or is not responding"
    return 1
}

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "📍 Working from: $SCRIPT_DIR"

# Check if services are already running
echo "🔍 Checking if services are already running..."

spam_running=false
autocomplete_running=false

if check_service 8082 "Spam Detection Service"; then
    spam_running=true
fi

if check_service 8000 "Autocomplete Service"; then
    autocomplete_running=true
fi

# Start services that aren't running
if [ "$spam_running" = false ]; then
    spam_dir="$SCRIPT_DIR/spam_detection"
    if [ -d "$spam_dir" ]; then
        if [ -f "$spam_dir/venv/bin/python" ]; then
            start_service "$spam_dir" "./venv/bin/python src/server.py" "Spam Detection Service" 8082
        else
            echo "❌ Virtual environment not found for spam detection service"
            echo "   Please run: cd $spam_dir && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
        fi
    else
        echo "❌ Spam detection directory not found: $spam_dir"
    fi
fi

if [ "$autocomplete_running" = false ]; then
    autocomplete_dir="$SCRIPT_DIR/autocomplete"
    if [ -d "$autocomplete_dir" ]; then
        # Check for Python environment
        if command -v python3 > /dev/null 2>&1; then
            start_service "$autocomplete_dir" "python3 api_server.py" "Autocomplete Service" 8000
        else
            echo "❌ Python3 not found for autocomplete service"
        fi
    else
        echo "❌ Autocomplete directory not found: $autocomplete_dir"
    fi
fi

echo ""
echo "🎯 Service Status Summary:"
echo "========================="

# Final status check
if check_service 8082 "Spam Detection Service"; then
    echo "✅ Spam Detection: http://localhost:8082"
    echo "   📖 API Docs: http://localhost:8082/docs"
else
    echo "❌ Spam Detection: Failed to start"
fi

if check_service 8000 "Autocomplete Service"; then
    echo "✅ Autocomplete: http://localhost:8000"
    echo "   📖 API Docs: http://localhost:8000/docs"
else
    echo "❌ Autocomplete: Failed to start"
fi

echo ""
echo "📋 Next Steps:"
echo "1. Run the integration test: python test_ai_integration.py"
echo "2. Start the Elixir backend to complete the integration"
echo "3. Check service logs if any issues occur:"
echo "   - Spam Detection: spam_detection/spam_detection_service.log"
echo "   - Autocomplete: autocomplete/autocomplete_service.log"

echo ""
echo "🛑 To stop services later, use:"
echo "   pkill -f 'python.*server.py'"
echo "   pkill -f 'python.*api_server.py'"
