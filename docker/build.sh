#!/bin/bash

# Build script for Docker images
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Sup Docker Images${NC}"
echo "=================================="

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default values
REGISTRY=""
TAG="latest"
PUSH=false
BUILD_ALL=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        --backend-only)
            BUILD_ALL=false
            BUILD_BACKEND=true
            shift
            ;;
        --frontend-only)
            BUILD_ALL=false
            BUILD_FRONTEND=true
            shift
            ;;
        --autocomplete-only)
            BUILD_ALL=false
            BUILD_AUTOCOMPLETE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -r, --registry REGISTRY    Docker registry to use"
            echo "  -t, --tag TAG              Tag for the images (default: latest)"
            echo "  -p, --push                 Push images to registry"
            echo "  --backend-only             Build only backend image"
            echo "  --frontend-only            Build only frontend image"
            echo "  --autocomplete-only        Build only autocomplete image"
            echo "  -h, --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Build functions
build_backend() {
    echo -e "${YELLOW}Building backend image...${NC}"
    local image_name="${REGISTRY:+$REGISTRY/}sup/backend:$TAG"
    docker build -f docker/backend.Dockerfile -t "$image_name" ./backend
    echo -e "${GREEN}✓ Backend image built: $image_name${NC}"
    
    if [ "$PUSH" = true ]; then
        echo -e "${YELLOW}Pushing backend image...${NC}"
        docker push "$image_name"
        echo -e "${GREEN}✓ Backend image pushed${NC}"
    fi
}

build_frontend() {
    echo -e "${YELLOW}Building frontend image...${NC}"
    local image_name="${REGISTRY:+$REGISTRY/}sup/frontend:$TAG"
    docker build -f docker/frontend.Dockerfile -t "$image_name" ./frontend
    echo -e "${GREEN}✓ Frontend image built: $image_name${NC}"
    
    if [ "$PUSH" = true ]; then
        echo -e "${YELLOW}Pushing frontend image...${NC}"
        docker push "$image_name"
        echo -e "${GREEN}✓ Frontend image pushed${NC}"
    fi
}

build_autocomplete() {
    echo -e "${YELLOW}Building autocomplete image...${NC}"
    local image_name="${REGISTRY:+$REGISTRY/}sup/autocomplete:$TAG"
    docker build -t "$image_name" ./autocomplete
    echo -e "${GREEN}✓ Autocomplete image built: $image_name${NC}"
    
    if [ "$PUSH" = true ]; then
        echo -e "${YELLOW}Pushing autocomplete image...${NC}"
        docker push "$image_name"
        echo -e "${GREEN}✓ Autocomplete image pushed${NC}"
    fi
}

# Main build logic
if [ "$BUILD_ALL" = true ]; then
    build_backend
    build_frontend
    build_autocomplete
else
    [ "$BUILD_BACKEND" = true ] && build_backend
    [ "$BUILD_FRONTEND" = true ] && build_frontend
    [ "$BUILD_AUTOCOMPLETE" = true ] && build_autocomplete
fi

echo -e "${GREEN}All builds completed successfully!${NC}"

# Show built images
echo -e "\n${YELLOW}Built images:${NC}"
docker images | grep -E "(sup/backend|sup/frontend|sup/autocomplete)" | head -10
