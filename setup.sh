#!/bin/bash

# Sup Messaging App - Development Setup Script
echo "🚀 Setting up Sup Messaging App for development..."

# Check prerequisites
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo "❌ $1 is not installed. Please install it first."
        exit 1
    else
        echo "✅ $1 is available"
    fi
}

echo "📋 Checking prerequisites..."
check_command "elixir"
check_command "mix"
check_command "node"
check_command "npm"
check_command "psql"
check_command "redis-cli"

# Setup backend
echo "🔧 Setting up backend..."
cd backend

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created .env file from template"
fi

# Install Elixir dependencies
echo "📦 Installing Elixir dependencies..."
mix deps.get

# Setup database
echo "🗄️  Setting up database..."
mix ecto.create
mix ecto.migrate

echo "✅ Backend setup complete!"

# Setup frontend
echo "🔧 Setting up frontend..."
cd ../frontend

# Install Node.js dependencies
echo "📦 Installing Node.js dependencies..."
npm install

echo "✅ Frontend setup complete!"

# Return to root directory
cd ..

echo "🎉 Development setup complete!"
echo ""
echo "🚀 To start the application:"
echo "Backend:  cd backend && mix run --no-halt"
echo "Frontend: cd frontend && npm start"
echo ""
echo "🐳 Or use Docker: docker-compose up -d"
echo ""
echo "📱 Access points:"
echo "- Backend API: http://localhost:4000"
echo "- Frontend Web: http://localhost:19006"
echo "- API Health: http://localhost:4000/health"
