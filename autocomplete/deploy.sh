#!/bin/bash

# Autocomplete Microservice Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
SERVICE_NAME="autocomplete"
DOCKER_COMPOSE_FILE="docker-compose.yml"
HEALTH_CHECK_URL="http://localhost:8000/health"
MAX_WAIT_TIME=120

echo -e "${GREEN}🚀 Deploying Autocomplete Microservice${NC}"
echo "=================================================="

# Function to check if service is healthy
check_health() {
    echo -e "${YELLOW}🔍 Checking service health...${NC}"
    
    for i in $(seq 1 $MAX_WAIT_TIME); do
        if curl -s $HEALTH_CHECK_URL > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Service is healthy!${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    
    echo -e "${RED}❌ Service health check failed after ${MAX_WAIT_TIME}s${NC}"
    return 1
}

# Function to show service logs
show_logs() {
    echo -e "${YELLOW}📋 Recent service logs:${NC}"
    docker-compose -f $DOCKER_COMPOSE_FILE logs --tail=20 $SERVICE_NAME
}

# Parse command line arguments
COMMAND=${1:-"deploy"}

case $COMMAND in
    "deploy")
        echo -e "${YELLOW}🔨 Building and starting services...${NC}"
        
        # Build and start services
        docker-compose -f $DOCKER_COMPOSE_FILE build
        docker-compose -f $DOCKER_COMPOSE_FILE up -d
        
        # Wait for service to be healthy
        if check_health; then
            echo -e "${GREEN}🎉 Deployment successful!${NC}"
            echo -e "${GREEN}📍 Service available at: $HEALTH_CHECK_URL${NC}"
        else
            echo -e "${RED}💥 Deployment failed!${NC}"
            show_logs
            exit 1
        fi
        ;;
        
    "stop")
        echo -e "${YELLOW}🛑 Stopping services...${NC}"
        docker-compose -f $DOCKER_COMPOSE_FILE down
        echo -e "${GREEN}✅ Services stopped${NC}"
        ;;
        
    "restart")
        echo -e "${YELLOW}🔄 Restarting services...${NC}"
        docker-compose -f $DOCKER_COMPOSE_FILE down
        docker-compose -f $DOCKER_COMPOSE_FILE up -d
        
        if check_health; then
            echo -e "${GREEN}✅ Restart successful!${NC}"
        else
            echo -e "${RED}❌ Restart failed!${NC}"
            show_logs
            exit 1
        fi
        ;;
        
    "logs")
        show_logs
        ;;
        
    "status")
        echo -e "${YELLOW}📊 Service status:${NC}"
        docker-compose -f $DOCKER_COMPOSE_FILE ps
        
        echo -e "\n${YELLOW}🔍 Health check:${NC}"
        if curl -s $HEALTH_CHECK_URL | python -m json.tool 2>/dev/null; then
            echo -e "${GREEN}✅ Service is responding${NC}"
        else
            echo -e "${RED}❌ Service is not responding${NC}"
        fi
        ;;
        
    "dev")
        echo -e "${YELLOW}🔧 Starting development environment...${NC}"
        docker-compose -f docker-compose.dev.yml up --build
        ;;
        
    "clean")
        echo -e "${YELLOW}🧹 Cleaning up...${NC}"
        docker-compose -f $DOCKER_COMPOSE_FILE down -v
        docker-compose -f docker-compose.dev.yml down -v 2>/dev/null || true
        docker system prune -f
        echo -e "${GREEN}✅ Cleanup complete${NC}"
        ;;
        
    *)
        echo -e "${RED}❌ Unknown command: $COMMAND${NC}"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  deploy    - Build and deploy the service (default)"
        echo "  stop      - Stop the service"
        echo "  restart   - Restart the service"
        echo "  logs      - Show service logs"
        echo "  status    - Show service status and health"
        echo "  dev       - Start development environment"
        echo "  clean     - Clean up containers and volumes"
        exit 1
        ;;
esac