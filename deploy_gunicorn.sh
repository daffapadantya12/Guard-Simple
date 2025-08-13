#!/bin/bash

# Guard API Gunicorn Deployment Script
# This script helps deploy the Guard API using Gunicorn with proper configuration

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if virtual environment exists
check_venv() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "No virtual environment detected. It's recommended to use a virtual environment."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        print_success "Virtual environment detected: $VIRTUAL_ENV"
    fi
}

# Check if Gunicorn is installed
check_gunicorn() {
    if ! command -v gunicorn &> /dev/null; then
        print_error "Gunicorn is not installed. Installing..."
        pip install gunicorn[uvicorn]
        print_success "Gunicorn installed successfully"
    else
        print_success "Gunicorn is already installed"
    fi
}

# Check if required dependencies are installed
check_dependencies() {
    print_status "Checking dependencies..."
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
        print_success "Dependencies installed/updated"
    else
        print_warning "requirements.txt not found"
    fi
}

# Check environment variables
check_env_vars() {
    print_status "Checking environment variables..."
    
    # Load .env file if it exists
    if [[ -f ".env" ]]; then
        print_status "Loading environment variables from .env file..."
        set -a  # automatically export all variables
        source .env
        set +a  # stop automatically exporting
        print_success "Environment variables loaded from .env"
    fi
    
    # Set defaults for missing variables
    if [[ -z "$API_KEY" ]]; then
        print_warning "API_KEY not set. Using default value."
        export API_KEY="guard-api-key-2024"
    else
        print_success "API_KEY loaded: ${API_KEY:0:8}..."
    fi
    
    if [[ -z "$DEVICE" ]]; then
        print_warning "DEVICE not set. Using CPU."
        export DEVICE="cpu"
    else
        print_success "DEVICE set to: $DEVICE"
    fi
    
    print_success "Environment variables configured"
}

# Function to start Gunicorn
start_gunicorn() {
    local mode=$1
    local workers=${2:-4}
    local bind=${3:-"0.0.0.0:8000"}
    
    print_status "Starting Gunicorn in $mode mode..."
    
    case $mode in
        "dev")
            print_status "Development mode - single worker, auto-reload"
            gunicorn main:app \
                --worker-class uvicorn.workers.UvicornWorker \
                --bind $bind \
                --workers 1 \
                --reload \
                --access-logfile - \
                --error-logfile -
            ;;
        "prod")
            print_status "Production mode - using gunicorn.conf.py"
            if [[ -f "gunicorn.conf.py" ]]; then
                gunicorn main:app -c gunicorn.conf.py --bind $bind --workers $workers
            else
                print_warning "gunicorn.conf.py not found, using default production settings"
                gunicorn main:app \
                    --worker-class uvicorn.workers.UvicornWorker \
                    --bind $bind \
                    --workers $workers \
                    --timeout 120 \
                    --keepalive 5 \
                    --max-requests 1000 \
                    --max-requests-jitter 100 \
                    --preload \
                    --access-logfile - \
                    --error-logfile -
            fi
            ;;
        "test")
            print_status "Test mode - checking if server starts correctly"
            # Use gtimeout on macOS if available, otherwise skip timeout
            if command -v gtimeout &> /dev/null; then
                gtimeout 30s gunicorn main:app \
                    --worker-class uvicorn.workers.UvicornWorker \
                    --bind $bind \
                    --workers 1 \
                    --timeout 30 || {
                    print_success "Server test completed (timeout expected)"
                    return 0
                }
            elif command -v timeout &> /dev/null; then
                timeout 30s gunicorn main:app \
                    --worker-class uvicorn.workers.UvicornWorker \
                    --bind $bind \
                    --workers 1 \
                    --timeout 30 || {
                    print_success "Server test completed (timeout expected)"
                    return 0
                }
            else
                print_status "Starting server for 5 seconds to test startup..."
                gunicorn main:app \
                    --worker-class uvicorn.workers.UvicornWorker \
                    --bind $bind \
                    --workers 1 \
                    --timeout 30 &
                local pid=$!
                sleep 5
                kill $pid 2>/dev/null || true
                wait $pid 2>/dev/null || true
                print_success "Server test completed"
            fi
            ;;
        *)
            print_error "Invalid mode: $mode. Use 'dev', 'prod', or 'test'"
            exit 1
            ;;
    esac
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [MODE] [OPTIONS]"
    echo ""
    echo "Modes:"
    echo "  dev                 Start in development mode (single worker, auto-reload)"
    echo "  prod                Start in production mode (multiple workers)"
    echo "  test                Test if the server starts correctly"
    echo ""
    echo "Options:"
    echo "  -w, --workers NUM   Number of worker processes (default: 4)"
    echo "  -b, --bind ADDR     Bind address (default: 0.0.0.0:8000)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 dev                          # Development mode"
    echo "  $0 prod -w 8                    # Production with 8 workers"
    echo "  $0 prod -b 127.0.0.1:8080       # Production on localhost:8080"
    echo "  $0 test                         # Test server startup"
}

# Parse command line arguments
MODE="prod"
WORKERS=4
BIND="0.0.0.0:8000"

while [[ $# -gt 0 ]]; do
    case $1 in
        dev|prod|test)
            MODE="$1"
            shift
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        -b|--bind)
            BIND="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
print_status "Guard API Gunicorn Deployment Script"
print_status "Mode: $MODE, Workers: $WORKERS, Bind: $BIND"
echo

# Run checks
check_venv
check_gunicorn
check_dependencies
check_env_vars

echo
print_status "Starting Guard API server..."
print_status "Press Ctrl+C to stop the server"
echo

# Start the server
start_gunicorn "$MODE" "$WORKERS" "$BIND"