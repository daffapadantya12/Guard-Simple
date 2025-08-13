# Guard API Deployment Examples

This document provides practical examples for deploying the Guard API using Gunicorn in different scenarios.

## Quick Start

### 1. Development Mode
For local development with auto-reload:
```bash
./deploy_gunicorn.sh dev
```

### 2. Production Mode
For production deployment:
```bash
./deploy_gunicorn.sh prod
```

### 3. Test Server Startup
To verify the server starts correctly:
```bash
./deploy_gunicorn.sh test
```

## Advanced Examples

### Custom Worker Count
```bash
# Use 8 workers for high-traffic scenarios
./deploy_gunicorn.sh prod --workers 8
```

### Custom Bind Address
```bash
# Bind to localhost only
./deploy_gunicorn.sh prod --bind 127.0.0.1:8080

# Bind to specific interface
./deploy_gunicorn.sh prod --bind 192.168.1.100:8000
```

### Combined Options
```bash
# Production with 6 workers on port 9000
./deploy_gunicorn.sh prod -w 6 -b 0.0.0.0:9000
```

## Environment Setup

### Set Environment Variables
```bash
# Set API key
export API_KEY="your-secure-production-key"

# Set device (cpu/cuda)
export DEVICE="cuda"

# Set Redis URL
export REDIS_URL="redis://localhost:6379"

# Then start the server
./deploy_gunicorn.sh prod
```

### Using .env File
Create a `.env` file:
```env
API_KEY=your-secure-production-key
DEVICE=cuda
REDIS_URL=redis://localhost:6379
HF_TOKEN=your-huggingface-token
```

## Docker Deployment

### Build and Run
```bash
# Build the Docker image
docker build -t guard-api .

# Run with environment variables
docker run -d \
  --name guard-api \
  -p 8000:8000 \
  -e API_KEY="production-key" \
  -e DEVICE="cuda" \
  --gpus all \
  guard-api
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  guard-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - API_KEY=production-key
      - DEVICE=cuda
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

Run with:
```bash
docker-compose up -d
```

## Production Considerations

### Resource Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ (4-8GB per worker for AI models)
- **GPU**: Optional, but recommended for faster inference
- **Storage**: 10GB+ for model cache

### Performance Tuning
```bash
# For CPU-intensive workloads
./deploy_gunicorn.sh prod -w $(($(nproc) * 2 + 1))

# For GPU workloads (limit workers to avoid GPU memory issues)
./deploy_gunicorn.sh prod -w 2

# For high-memory systems
./deploy_gunicorn.sh prod -w 8 -b 0.0.0.0:8000
```

### Monitoring
```bash
# Check server status
curl http://localhost:8000/healthz

# Monitor processes
ps aux | grep gunicorn

# Check logs
tail -f /var/log/guard-api/access.log
```

## Troubleshooting

### Common Issues

1. **Port already in use**:
   ```bash
   # Find process using port 8000
   lsof -i :8000
   # Kill the process
   kill -9 <PID>
   # Or use different port
   ./deploy_gunicorn.sh prod -b 0.0.0.0:8001
   ```

2. **Permission denied**:
   ```bash
   chmod +x deploy_gunicorn.sh
   ```

3. **Module not found errors**:
   ```bash
   # Activate virtual environment
   source venv/bin/activate
   # Install dependencies
   pip install -r requirements.txt
   ```

4. **Memory issues**:
   ```bash
   # Reduce workers
   ./deploy_gunicorn.sh prod -w 2
   ```

5. **GPU memory errors**:
   ```bash
   # Use CPU mode
   export DEVICE=cpu
   ./deploy_gunicorn.sh prod
   ```

6. **Model loading timeout**:
   - Increase timeout in `gunicorn.conf.py`
   - Use faster storage (SSD)
   - Pre-download models

7. **macOS `/dev/shm` error**:
   - **Issue**: `RuntimeError: /dev/shm doesn't exist. Can't create workertmp.`
   - **Solution**: The `gunicorn.conf.py` has been updated to use `/tmp` instead of `/dev/shm` for cross-platform compatibility
   - **Note**: This is automatically handled in the current configuration

### Health Checks
```bash
# Basic health check
curl -f http://localhost:8000/healthz || exit 1

# API authentication check
curl -H "Authorization: Bearer your-api-key" \
     http://localhost:8000/config
```

## Security Best Practices

1. **Use strong API keys**:
   ```bash
   export API_KEY=$(openssl rand -hex 32)
   ```

2. **Enable HTTPS** (update `gunicorn.conf.py`):
   ```python
   keyfile = "/path/to/private.key"
   certfile = "/path/to/certificate.crt"
   ```

3. **Use reverse proxy** (Nginx/Apache)
4. **Firewall configuration**
5. **Regular security updates**

## Scaling

### Horizontal Scaling
```bash
# Multiple instances on different ports
./deploy_gunicorn.sh prod -b 0.0.0.0:8000 &
./deploy_gunicorn.sh prod -b 0.0.0.0:8001 &
./deploy_gunicorn.sh prod -b 0.0.0.0:8002 &
```

### Load Balancer Configuration
See the main README.md for Nginx configuration examples.