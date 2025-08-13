# Docker Production Deployment Guide

This guide explains how to deploy the Guard API system using Docker in a production environment.

## 🚀 Quick Start

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB+ RAM (for AI models)
- 10GB+ disk space

### 1. Clone and Setup
```bash
git clone https://github.com/daffapadantya12/Guard-Simple.git
cd Guard-Simple
```

### 2. Configure Environment
```bash
# Copy the production environment template
cp .env.production .env

# Edit the environment file with your settings
nano .env
```

**Required Configuration:**
- `HUGGINGFACE_TOKEN`: Get from [Hugging Face Settings](https://huggingface.co/settings/tokens)
- `CORS_ORIGINS`: Update with your domain(s)
- `REDIS_URL`: Keep as `redis://redis:6379` for Docker

### 3. Deploy
```bash
# Full deployment (build + start)
./deploy.sh deploy

# Or step by step:
./deploy.sh build
./deploy.sh up
```

### 4. Verify Deployment
```bash
# Check service status
./deploy.sh status

# View logs
./deploy.sh logs

# Test API
curl http://localhost:8000/healthz
```

## 📋 Available Commands

| Command | Description |
|---------|-------------|
| `./deploy.sh deploy` | Full deployment (build + start) |
| `./deploy.sh build` | Build Docker images |
| `./deploy.sh up` | Start services |
| `./deploy.sh down` | Stop services |
| `./deploy.sh restart` | Restart services |
| `./deploy.sh logs` | Show service logs |
| `./deploy.sh status` | Show service status |
| `./deploy.sh clean` | Clean up containers and images |

## 🏗️ Architecture

### Services
- **guard-api**: FastAPI application with AI models
- **redis**: Redis cache and rate limiting storage

### Networking
- Internal network: `guard-network`
- External ports: 8000 (API), 6379 (Redis)

### Volumes
- `redis_data`: Persistent Redis data
- `./models_cache`: AI model cache (mounted from host)

## 🔧 Configuration

### Environment Variables

#### Core Settings
```bash
REDIS_URL=redis://redis:6379          # Redis connection
HUGGINGFACE_TOKEN=hf_xxx              # HF API token
DEVICE=auto                           # cuda/cpu/auto
```

#### Performance Tuning
```bash
USE_QUANTIZATION=true                 # Reduce memory usage
QUANTIZATION_BITS=4                   # 4-bit quantization
WORKERS=4                             # Gunicorn workers
CACHE_TTL=3600                        # Cache timeout (seconds)
```

#### Security
```bash
API_KEY=your_secure_api_key_here      # API authentication key
CORS_ORIGINS=https://yourdomain.com   # Allowed origins
RATE_LIMIT_REQUESTS=50                # Requests per window
RATE_LIMIT_WINDOW=60                  # Rate limit window (seconds)
```

### Docker Compose Override

For custom configurations, create `docker-compose.override.yml`:

```yaml
version: '3.8'
services:
  guard-api:
    environment:
      - WORKERS=8
      - LOG_LEVEL=DEBUG
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
```

## 🔍 Monitoring

### Health Checks
```bash
# API health
curl http://localhost:8000/healthz

# Redis health
docker-compose exec redis redis-cli ping

# Service status
docker-compose ps
```

### Metrics
```bash
# System metrics
curl http://localhost:8000/metrics

# Configuration
curl http://localhost:8000/config
```

### Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f guard-api
docker-compose logs -f redis

# With timestamps
docker-compose logs -f -t
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Out of Memory
```bash
# Check memory usage
docker stats

# Enable quantization
USE_QUANTIZATION=true
QUANTIZATION_BITS=4
```

#### 2. Model Loading Fails
```bash
# Check HuggingFace token
echo $HUGGINGFACE_TOKEN

# Check model cache permissions
ls -la models_cache/

# Clear cache and rebuild
./deploy.sh clean
./deploy.sh deploy
```

#### 3. Redis Connection Issues
```bash
# Check Redis status
docker-compose exec redis redis-cli ping

# Check network connectivity
docker-compose exec guard-api ping redis
```

#### 4. API Not Responding
```bash
# Check container status
docker-compose ps

# Check logs for errors
docker-compose logs guard-api

# Restart services
./deploy.sh restart
```

### Performance Optimization

#### Memory Usage
- Enable quantization: `USE_QUANTIZATION=true`
- Use CPU if GPU memory is limited: `DEVICE=cpu`
- Reduce cache TTL: `CACHE_TTL=1800`

#### Response Time
- Increase workers: `WORKERS=8`
- Use GPU if available: `DEVICE=cuda`
- Optimize model cache location

## 🔒 Security Best Practices

### Production Checklist
- [ ] Use strong HuggingFace token
- [ ] Configure proper CORS origins
- [ ] Set appropriate rate limits
- [ ] Use HTTPS in production
- [ ] Regular security updates
- [ ] Monitor logs for suspicious activity

### Network Security
```yaml
# docker-compose.override.yml
services:
  redis:
    ports: []  # Remove external Redis access
  guard-api:
    ports:
      - "127.0.0.1:8000:8000"  # Bind to localhost only
```

## 📊 Scaling

### Horizontal Scaling
```yaml
# docker-compose.override.yml
services:
  guard-api:
    deploy:
      replicas: 3
    ports:
      - "8000-8002:8000"
```

### Load Balancing
Use nginx or similar:
```nginx
upstream guard_api {
    server localhost:8000;
    server localhost:8001;
    server localhost:8002;
}

server {
    listen 80;
    location / {
        proxy_pass http://guard_api;
    }
}
```

## 🔄 Updates and Maintenance

### Updating the Application
```bash
# Pull latest changes
git pull

# Rebuild and restart
./deploy.sh clean
./deploy.sh deploy
```

### Backup
```bash
# Backup Redis data
docker-compose exec redis redis-cli BGSAVE
cp -r redis_data/ backup/redis_$(date +%Y%m%d)/

# Backup model cache
tar -czf backup/models_$(date +%Y%m%d).tar.gz models_cache/
```

### Maintenance
```bash
# Clean up old images
docker image prune -f

# Clean up old containers
docker container prune -f

# Full system cleanup
docker system prune -af
```

## 📞 Support

For issues:
1. Check logs: `./deploy.sh logs`
2. Verify health: `./deploy.sh status`
3. Review configuration: `.env` file
4. Check system resources: `docker stats`

## 🎯 API Endpoints

Once deployed, the API will be available at:

### Public Endpoints (No Authentication)
- **Health Check**: `GET http://localhost:8000/healthz`
- **API Documentation**: `GET http://localhost:8000/docs`

### Protected Endpoints (Require API Key)
- **Analyze Prompt**: `POST http://localhost:8000/analyze`
- **Metrics**: `GET http://localhost:8000/metrics`
- **Configuration**: `GET http://localhost:8000/config`
- **Clear Cache**: `DELETE http://localhost:8000/cache`

### Authentication
All protected endpoints require a Bearer token in the Authorization header:

```bash
# Set your API key
export API_KEY="your_secure_api_key_here"

# Analyze a prompt
curl -X POST "http://localhost:8000/analyze" \
     -H "Authorization: Bearer $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello world", "user_id": "test", "lang": "auto"}'

# Get configuration
curl -X GET "http://localhost:8000/config" \
     -H "Authorization: Bearer $API_KEY"

# Health check (no auth required)
curl -X GET "http://localhost:8000/healthz"
```

### API Key Configuration
- Set `API_KEY` environment variable in your `.env` file
- Default key for development: `guard-api-key-2024-secure`
- **Important**: Change the API key in production for security