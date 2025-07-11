#!/bin/bash

# setup_secure.sh
# Secure setup script for the pavement classification model

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if running as root (security check)
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons"
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required but not installed"
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ $(echo "$python_version < 3.8" | bc -l) -eq 1 ]]; then
        error "Python 3.8 or higher is required. Found: $python_version"
        exit 1
    fi
    
    success "Python $python_version found"
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        error "pip3 is required but not installed"
        exit 1
    fi
    
    # Check disk space (require at least 10GB)
    available_space=$(df . | tail -1 | awk '{print $4}')
    required_space=10485760  # 10GB in KB
    
    if [[ $available_space -lt $required_space ]]; then
        warning "Low disk space. Available: $(($available_space / 1024 / 1024))GB, Recommended: 10GB+"
    fi
    
    success "System requirements check passed"
}

# Setup virtual environment
setup_venv() {
    log "Setting up virtual environment..."
    
    if [[ -d "venv" ]]; then
        warning "Virtual environment already exists. Removing..."
        rm -rf venv
    fi
    
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    success "Virtual environment created and activated"
}

# Install dependencies
install_dependencies() {
    log "Installing dependencies..."
    
    if [[ ! -f "requirements.txt" ]]; then
        error "requirements.txt not found"
        exit 1
    fi
    
    # Install dependencies with security considerations
    pip install --no-cache-dir --require-hashes --only-binary=all -r requirements.txt 2>/dev/null || {
        warning "Secure installation failed, falling back to standard installation"
        pip install -r requirements.txt
    }
    
    success "Dependencies installed"
}

# Setup environment file
setup_environment() {
    log "Setting up environment configuration..."
    
    if [[ ! -f ".env.example" ]]; then
        error ".env.example not found"
        exit 1
    fi
    
    if [[ -f ".env" ]]; then
        warning ".env file already exists"
        read -p "Do you want to overwrite it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Skipping environment setup"
            return
        fi
    fi
    
    cp .env.example .env
    
    # Set secure permissions on .env file
    chmod 600 .env
    
    warning "Please edit .env file with your actual credentials"
    warning "NEVER commit the .env file to version control"
    
    success "Environment file created with secure permissions"
}

# Validate configuration
validate_config() {
    log "Validating configuration..."
    
    if [[ ! -f ".env" ]]; then
        error ".env file not found. Please run setup first."
        exit 1
    fi
    
    # Source environment variables
    set -a
    source .env
    set +a
    
    # Check required variables
    required_vars=("AWS_ACCESS_KEY_ID" "AWS_SECRET_ACCESS_KEY" "S3_BUCKET_NAME" "MLFLOW_TRACKING_URI")
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    # Test configuration
    python3 config.py || {
        error "Configuration validation failed"
        exit 1
    }
    
    success "Configuration validation passed"
}

# Setup logging directory
setup_logging() {
    log "Setting up logging..."
    
    mkdir -p logs
    chmod 755 logs
    
    success "Logging directory created"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    directories=("checkpoints" "artifacts" "logs")
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
        chmod 755 "$dir"
    done
    
    success "Directories created"
}

# Security hardening
security_hardening() {
    log "Applying security hardening..."
    
    # Set secure permissions on Python files
    find . -name "*.py" -exec chmod 644 {} \;
    
    # Make main.py executable
    chmod 755 main.py
    
    # Secure shell scripts
    find . -name "*.sh" -exec chmod 755 {} \;
    
    # Create .gitignore if it doesn't exist
    if [[ ! -f ".gitignore" ]]; then
        cat > .gitignore << EOF
# Environment and secrets
.env
*.env
.env.*

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environment
venv/
env/
ENV/

# Logs
*.log
logs/

# Checkpoints and models
checkpoints/
*.pth
*.pkl

# MLflow
mlruns/
mlflow-artifacts/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOF
        success "Created .gitignore file"
    fi
    
    success "Security hardening applied"
}

# Main setup function
main() {
    log "Starting secure setup for Pavement Classification Model"
    log "=================================================="
    
    check_root
    check_requirements
    setup_venv
    install_dependencies
    setup_environment
    setup_logging
    create_directories
    security_hardening
    
    log "=================================================="
    success "Setup completed successfully!"
    log ""
    log "Next steps:"
    log "1. Edit .env file with your AWS credentials and MLflow URI"
    log "2. Activate virtual environment: source venv/bin/activate"
    log "3. Validate configuration: python3 config.py"
    log "4. Run training: python3 main.py"
    log ""
    warning "Remember to NEVER commit .env file to version control"
}

# Run main function
main "$@"
