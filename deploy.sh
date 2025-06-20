#!/bin/bash

# Sup - Complete Deployment Script
# This script handles the complete deployment of the Sup messaging application

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on supported OS
check_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    print_status "Detected OS: $OS"
}

# Check if required tools are installed
check_dependencies() {
    print_status "Checking dependencies..."
    
    local missing_deps=()
    
    # Check for Docker
    if ! command -v docker &> /dev/null; then
        missing_deps+=("docker")
    fi
    
    # Check for Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        missing_deps+=("docker-compose")
    fi
    
    # Check for kubectl (optional for K8s deployment)
    if [ "$DEPLOYMENT_MODE" = "kubernetes" ]; then
        if ! command -v kubectl &> /dev/null; then
            missing_deps+=("kubectl")
        fi
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing dependencies: ${missing_deps[*]}"
        print_status "Please install the missing dependencies and run again"
        exit 1
    fi
    
    print_success "All dependencies are installed"
}

# Build Docker images
build_images() {
    print_status "Building Docker images..."
    
    # Build backend image
    print_status "Building backend image..."
    docker build -t sup/backend:latest -f backend/Dockerfile backend/
    
    # Build frontend image
    print_status "Building frontend image..."
    docker build -t sup/frontend:latest -f frontend/Dockerfile frontend/
    
    # Build autocomplete service image
    print_status "Building autocomplete service image..."
    docker build -t sup/autocomplete:latest -f autocomplete/Dockerfile autocomplete/
    
    print_success "All images built successfully"
}

# Deploy using Docker Compose
deploy_docker_compose() {
    print_status "Deploying with Docker Compose..."
    
    # Copy environment template if it doesn't exist
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            print_warning "Created .env file from template. Please review and update it."
        else
            print_warning "No .env file found. Using default values."
        fi
    fi
    
    # Start services
    docker-compose -f docker-compose.yml up -d
    
    # Wait for services to be ready
    print_status "Waiting for services to start..."
    sleep 30
    
    # Run database migrations
    print_status "Running database migrations..."
    docker-compose exec backend mix ecto.migrate
    
    print_success "Docker Compose deployment completed"
    print_status "Frontend available at: http://localhost:19006"
    print_status "Backend API available at: http://localhost:4000"
    print_status "Autocomplete service available at: http://localhost:8000"
}

# Deploy using Kubernetes
deploy_kubernetes() {
    print_status "Deploying with Kubernetes..."
    
    # Create namespace
    kubectl apply -f k8s/namespace.yaml
    
    # Deploy databases first
    print_status "Deploying databases..."
    kubectl apply -f k8s/postgres-deployment.yaml
    kubectl apply -f k8s/scylladb-deployment.yaml
    kubectl apply -f k8s/redis-deployment.yaml
    
    # Wait for databases to be ready
    print_status "Waiting for databases to be ready..."
    kubectl wait --for=condition=ready pod -l app=postgres -n sup --timeout=300s
    kubectl wait --for=condition=ready pod -l app=redis -n sup --timeout=300s
    kubectl wait --for=condition=ready pod -l app=scylladb -n sup --timeout=300s
    
    # Deploy autocomplete service
    print_status "Deploying autocomplete service..."
    kubectl apply -f k8s/autocomplete-deployment.yaml
    kubectl wait --for=condition=ready pod -l app=autocomplete -n sup --timeout=300s
    
    # Deploy backend
    print_status "Deploying backend..."
    kubectl apply -f k8s/backend-deployment.yaml
    kubectl wait --for=condition=ready pod -l app=backend -n sup --timeout=300s
    
    # Run database migrations
    print_status "Running database migrations..."
    kubectl exec -n sup deployment/backend -- mix ecto.migrate
    
    # Deploy frontend
    print_status "Deploying frontend..."
    kubectl apply -f k8s/frontend-deployment.yaml
    kubectl wait --for=condition=ready pod -l app=frontend -n sup --timeout=300s
    
    # Apply monitoring and policies
    print_status "Applying monitoring and policies..."
    kubectl apply -f k8s/monitoring-and-policies.yaml
    kubectl apply -f k8s/hpa.yaml
    
    print_success "Kubernetes deployment completed"
    print_status "Run 'kubectl get pods -n sup' to check pod status"
    print_status "Run 'kubectl get services -n sup' to see service endpoints"
}

# Deploy development environment
deploy_development() {
    print_status "Setting up development environment..."
    
    # Start development services
    docker-compose -f docker-compose.dev.yml up -d postgres redis scylladb
    
    # Wait for databases
    print_status "Waiting for databases..."
    sleep 20
    
    # Setup backend
    print_status "Setting up backend..."
    cd backend
    if [ ! -d "deps" ]; then
        mix deps.get
    fi
    mix ecto.create
    mix ecto.migrate
    cd ..
    
    # Setup frontend
    print_status "Setting up frontend..."
    cd frontend
    if [ ! -d "node_modules" ]; then
        npm install
    fi
    cd ..
    
    # Setup autocomplete service
    print_status "Setting up autocomplete service..."
    cd autocomplete
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    fi
    cd ..
    
    print_success "Development environment setup completed"
    print_status "To start the backend: cd backend && mix phx.server"
    print_status "To start the frontend: cd frontend && npm start"
    print_status "To start autocomplete service: cd autocomplete && ./activate.sh && python api_server.py"
}

# Clean up function
cleanup() {
    print_status "Cleaning up..."
    
    if [ "$DEPLOYMENT_MODE" = "docker" ]; then
        docker-compose down
    elif [ "$DEPLOYMENT_MODE" = "kubernetes" ]; then
        kubectl delete namespace sup
    elif [ "$DEPLOYMENT_MODE" = "development" ]; then
        docker-compose -f docker-compose.dev.yml down
    fi
    
    print_success "Cleanup completed"
}

# Health check function
health_check() {
    print_status "Running health checks..."
    
    local failed_checks=()
    
    # Check backend health
    if ! curl -f http://localhost:4000/health &> /dev/null; then
        failed_checks+=("backend")
    fi
    
    # Check autocomplete service health
    if ! curl -f http://localhost:8000/autocomplete/health &> /dev/null; then
        failed_checks+=("autocomplete")
    fi
    
    if [ ${#failed_checks[@]} -eq 0 ]; then
        print_success "All health checks passed"
    else
        print_warning "Failed health checks: ${failed_checks[*]}"
    fi
}

# Main deployment function
main() {
    print_status "Starting Sup deployment..."
    
    # Parse command line arguments
    DEPLOYMENT_MODE=${1:-"docker"}
    ACTION=${2:-"deploy"}
    
    case $ACTION in
        "deploy")
            check_os
            check_dependencies
            
            case $DEPLOYMENT_MODE in
                "docker")
                    build_images
                    deploy_docker_compose
                    ;;
                "kubernetes" | "k8s")
                    DEPLOYMENT_MODE="kubernetes"
                    build_images
                    deploy_kubernetes
                    ;;
                "development" | "dev")
                    DEPLOYMENT_MODE="development"
                    deploy_development
                    ;;
                *)
                    print_error "Unknown deployment mode: $DEPLOYMENT_MODE"
                    print_status "Available modes: docker, kubernetes, development"
                    exit 1
                    ;;
            esac
            
            # Run health checks after deployment
            sleep 10
            health_check
            ;;
        "cleanup")
            cleanup
            ;;
        "health")
            health_check
            ;;
        "build")
            build_images
            ;;
        *)
            print_error "Unknown action: $ACTION"
            print_status "Available actions: deploy, cleanup, health, build"
            exit 1
            ;;
    esac
    
    print_success "Deployment completed successfully!"
}

# Help function
show_help() {
    echo "Sup Deployment Script"
    echo ""
    echo "Usage: $0 [MODE] [ACTION]"
    echo ""
    echo "Deployment Modes:"
    echo "  docker      - Deploy using Docker Compose (default)"
    echo "  kubernetes  - Deploy using Kubernetes"
    echo "  development - Setup development environment"
    echo ""
    echo "Actions:"
    echo "  deploy      - Deploy the application (default)"
    echo "  cleanup     - Clean up deployed resources"
    echo "  health      - Run health checks"
    echo "  build       - Build Docker images only"
    echo ""
    echo "Examples:"
    echo "  $0                           # Deploy with Docker Compose"
    echo "  $0 kubernetes deploy         # Deploy with Kubernetes"
    echo "  $0 development deploy        # Setup development environment"
    echo "  $0 docker cleanup           # Clean up Docker deployment"
    echo "  $0 docker health            # Run health checks"
    echo ""
}

# Handle help request
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
fi

# Run main function
main "$@"
