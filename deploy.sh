#!/bin/bash

# Production deployment script for Guard API

set -e  # Exit on any error

echo "🚀 Starting Guard API Production Deployment"
echo "==========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install it and try again."
    exit 1
fi

# Function to show usage
show_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build     Build the Docker images"
    echo "  up        Start the services"
    echo "  down      Stop the services"
    echo "  restart   Restart the services"
    echo "  logs      Show service logs"
    echo "  status    Show service status"
    echo "  clean     Clean up containers and images"
    echo "  deploy    Full deployment (build + up)"
    echo ""
}

# Parse command line arguments
COMMAND=${1:-deploy}

case $COMMAND in
    "build")
        echo "🔨 Building Docker images..."
        docker-compose build --no-cache
        echo "✅ Build completed successfully!"
        ;;
    
    "up")
        echo "🚀 Starting services..."
        docker-compose up -d
        echo "✅ Services started successfully!"
        echo "📊 API available at: http://localhost:8000"
        echo "🔍 Health check: http://localhost:8000/healthz"
        echo "📚 API docs: http://localhost:8000/docs"
        ;;
    
    "down")
        echo "🛑 Stopping services..."
        docker-compose down
        echo "✅ Services stopped successfully!"
        ;;
    
    "restart")
        echo "🔄 Restarting services..."
        docker-compose down
        docker-compose up -d
        echo "✅ Services restarted successfully!"
        ;;
    
    "logs")
        echo "📋 Showing service logs..."
        docker-compose logs -f
        ;;
    
    "status")
        echo "📊 Service status:"
        docker-compose ps
        echo ""
        echo "🔍 Health checks:"
        echo "API Health: $(curl -s http://localhost:8000/healthz | jq -r '.status' 2>/dev/null || echo 'Not available')"
        echo "Redis Health: $(docker-compose exec redis redis-cli ping 2>/dev/null || echo 'Not available')"
        ;;
    
    "clean")
        echo "🧹 Cleaning up..."
        docker-compose down -v --remove-orphans
        docker system prune -f
        echo "✅ Cleanup completed!"
        ;;
    
    "deploy")
        echo "🚀 Full deployment starting..."
        
        # Build images
        echo "🔨 Building Docker images..."
        docker-compose build --no-cache
        
        # Start services
        echo "🚀 Starting services..."
        docker-compose up -d
        
        # Wait for services to be ready
        echo "⏳ Waiting for services to be ready..."
        sleep 10
        
        # Check health
        echo "🔍 Checking service health..."
        for i in {1..30}; do
            if curl -s http://localhost:8000/healthz > /dev/null; then
                echo "✅ API is healthy!"
                break
            fi
            echo "⏳ Waiting for API to be ready... ($i/30)"
            sleep 2
        done
        
        echo ""
        echo "🎉 Deployment completed successfully!"
        echo "==========================================="
        echo "📊 API available at: http://localhost:8000"
        echo "🔍 Health check: http://localhost:8000/healthz"
        echo "📚 API docs: http://localhost:8000/docs"
        echo "📋 View logs: ./deploy.sh logs"
        echo "📊 Check status: ./deploy.sh status"
        ;;
    
    "help" | "-h" | "--help")
        show_usage
        ;;
    
    *)
        echo "❌ Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac

echo ""
echo "💡 Tip: Use './deploy.sh help' to see all available commands"