#!/bin/bash

# Deployment script for Kubernetes
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Deploying Sup to Kubernetes${NC}"
echo "=================================="

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default values
NAMESPACE="sup"
DRY_RUN=false
SKIP_BUILD=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -n, --namespace NAMESPACE  Kubernetes namespace (default: sup)"
            echo "  --dry-run                  Show what would be deployed without applying"
            echo "  --skip-build               Skip building Docker images"
            echo "  -h, --help                 Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}kubectl is not installed or not in PATH${NC}"
    exit 1
fi

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}Cannot connect to Kubernetes cluster${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Kubernetes cluster is accessible${NC}"

# Build images if not skipped
if [ "$SKIP_BUILD" = false ]; then
    echo -e "${YELLOW}Building Docker images...${NC}"
    ./docker/build.sh
fi

# Deployment function
deploy_resources() {
    local dry_run_flag=""
    if [ "$DRY_RUN" = true ]; then
        dry_run_flag="--dry-run=client"
    fi

    echo -e "${YELLOW}Deploying Kubernetes resources...${NC}"
    
    # Apply namespace first
    kubectl apply $dry_run_flag -f k8s/namespace.yaml
    
    # Apply configurations and secrets
    kubectl apply $dry_run_flag -f k8s/postgres-deployment.yaml
    kubectl apply $dry_run_flag -f k8s/redis-deployment.yaml
    kubectl apply $dry_run_flag -f k8s/scylladb-deployment.yaml
    
    # Apply application deployments
    kubectl apply $dry_run_flag -f k8s/backend-deployment.yaml
    kubectl apply $dry_run_flag -f k8s/frontend-deployment.yaml
    kubectl apply $dry_run_flag -f k8s/autocomplete-deployment.yaml
    
    # Apply scaling and monitoring
    kubectl apply $dry_run_flag -f k8s/hpa.yaml
    kubectl apply $dry_run_flag -f k8s/monitoring-and-policies.yaml
    
    if [ "$DRY_RUN" = false ]; then
        echo -e "${GREEN}✓ All resources deployed${NC}"
        
        # Wait for deployments to be ready
        echo -e "${YELLOW}Waiting for deployments to be ready...${NC}"
        kubectl wait --for=condition=available --timeout=300s deployment/postgres -n $NAMESPACE
        kubectl wait --for=condition=available --timeout=300s deployment/redis -n $NAMESPACE
        kubectl wait --for=condition=available --timeout=300s deployment/backend -n $NAMESPACE
        kubectl wait --for=condition=available --timeout=300s deployment/frontend -n $NAMESPACE
        kubectl wait --for=condition=available --timeout=300s deployment/autocomplete -n $NAMESPACE
        
        echo -e "${GREEN}✓ All deployments are ready${NC}"
        
        # Show status
        echo -e "\n${YELLOW}Deployment Status:${NC}"
        kubectl get pods -n $NAMESPACE
        
        echo -e "\n${YELLOW}Services:${NC}"
        kubectl get services -n $NAMESPACE
        
        echo -e "\n${YELLOW}Ingresses:${NC}"
        kubectl get ingresses -n $NAMESPACE
        
        # Show access instructions
        echo -e "\n${GREEN}Access Instructions:${NC}"
        echo "Add the following entries to your /etc/hosts file:"
        echo "127.0.0.1 app.sup.local"
        echo "127.0.0.1 api.sup.local"
        echo "127.0.0.1 autocomplete.sup.local"
        echo ""
        echo "Then access the application at:"
        echo "Frontend: http://app.sup.local"
        echo "Backend API: http://api.sup.local"
        echo "Autocomplete: http://autocomplete.sup.local"
    else
        echo -e "${YELLOW}Dry run completed - no resources were actually deployed${NC}"
    fi
}

# Main deployment
deploy_resources

echo -e "${GREEN}Deployment completed successfully!${NC}"
