# GUARD-SIMPLE API - Complete Step-by-Step Tutorial

A comprehensive AI-powered prompt safety system that screens user inputs before sending them to LLMs. This tutorial will guide you through installation, setup, deployment, and usage of the GUARD-SIMPLE API.

## 🎯 What You'll Build

By following this tutorial, you'll deploy a production-ready prompt safety system featuring:
- **Multiple AI Safety Guards**: LLaMA Guard 8B/1B, IndoBERT Toxic Classifier, LLM Guard, NeMo Guardrails
- **Real-time Dashboard**: Monitor metrics, configure guards, and analyze prompts
- **Production Deployment**: Docker containerization with optimized performance
- **API Authentication**: Secure Bearer token authentication
- **Advanced Analytics**: Comprehensive metrics and threat intelligence

## 📋 Prerequisites

Before starting, ensure you have:
- **Python 3.11+** installed
- **Node.js 16+** installed
- **Docker** and **Docker Compose** installed
- **Redis server** (we'll show you how to install)
- **Hugging Face account** (free) for model access

## 🚀 Step 1: Project Setup

### 1.1 Clone the Repository

```bash
git clone https://github.com/your-username/GUARD-SIMPLE.git
cd GUARD-SIMPLE
```

### 1.2 Install System Dependencies

**On macOS:**
```bash
# Install Redis
brew install redis
brew services start redis

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install
```

**On Ubuntu/Debian:**
```bash
# Install Redis
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server

# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install
```

## 🔧 Step 2: Environment Configuration

### 2.1 Get Your Hugging Face Token

1. Visit [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Request access to LLaMA Guard models (optional but recommended)

### 2.2 Configure Environment Variables

**Option A: Interactive Setup (Recommended)**
```bash
python setup_env.py
```

**Option B: Manual Setup**
Create a `.env` file in the project root:

```bash
# Core Configuration
API_KEY=guard-api-key-2024
REACT_APP_API_KEY=guard-api-key-2024

# Hugging Face Configuration
HUGGINGFACE_TOKEN=your_token_here

# Device Configuration (auto/cuda/cpu)
DEVICE=auto

# Model Optimization (Recommended for production)
USE_QUANTIZATION=true
QUANTIZATION_BITS=4
QUANTIZATION_TYPE=nf4
USE_DOUBLE_QUANT=true
COMPUTE_DTYPE=float16
TORCH_DTYPE=float16

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Model Cache Directory
HF_CACHE_DIR=./models_cache

# Security
CORS_ORIGINS=["http://localhost:3000"]
LOG_LEVEL=INFO
```

## 🏃‍♂️ Step 3: Development Setup

### 3.1 Start the Backend API

```bash
# Start the FastAPI server
python main.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/healthz

### 3.2 Start the Frontend (New Terminal)

```bash
# In a new terminal window
npm start
```

The frontend will be available at:
- **Web Interface**: http://localhost:3000

### 3.3 Verify Installation

1. **Check API Health**:
   ```bash
   curl http://localhost:8000/healthz
   ```

2. **Test Authentication**:
   ```bash
   curl -H "Authorization: Bearer guard-api-key-2024" \
        http://localhost:8000/config
   ```

3. **Test Prompt Analysis**:
   ```bash
   curl -X POST "http://localhost:8000/analyze" \
        -H "Authorization: Bearer guard-api-key-2024" \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Hello, how are you?"}'
   ```

## 🐳 Step 4: Docker Deployment

### 4.1 Build Production Docker Image

```bash
# Build the production image
docker build -f Dockerfile.production -t tupaikapitalis/guard-api:latest .
```

### 4.2 Deploy with Docker Compose

```bash
# Deploy the application
docker-compose -f docker-compose.production.yml up -d
```

### 4.3 Verify Docker Deployment

```bash
# Check container status
docker ps

# View logs
docker logs <container_name>

# Test the deployed API
curl http://localhost:8000/healthz
```

### 4.4 Push to Docker Hub (Optional)

```bash
# Login to Docker Hub
docker login

# Push the image
docker push tupaikapitalis/guard-api:latest
```

## 🎛️ Step 5: Using the Web Interface

### 5.1 Access the Dashboard

Open http://localhost:3000 in your browser. You'll see three main tabs:

### 5.2 Input & Analysis Tab
- Enter prompts to test the safety guards
- View detailed analysis results from each guard
- See final verdict (Allow/Warn/Block)
- Monitor response times and guard performance

### 5.3 Dashboard Tab
- Real-time metrics and KPIs
- Guard performance statistics
- System health monitoring
- Historical data visualization

### 5.4 Settings Tab
- Toggle individual guards on/off
- Adjust sensitivity thresholds
- Configure system parameters
- View guard status and health

## 🔌 Step 6: API Integration

### 6.1 Basic Prompt Analysis

```python
import requests

api_key = "guard-api-key-2024"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Analyze a prompt
response = requests.post(
    "http://localhost:8000/analyze",
    headers=headers,
    json={
        "prompt": "Your prompt text here",
        "user_id": "user123",
        "lang": "auto"
    }
)

result = response.json()
print(f"Final verdict: {result['final']}")
print(f"Guard results: {result['per_guard']}")
```

### 6.2 Configuration Management

```python
# Get current configuration
config = requests.get(
    "http://localhost:8000/config",
    headers=headers
).json()

# Update configuration
requests.put(
    "http://localhost:8000/config",
    headers=headers,
    json={
        "enable_all": True,
        "guards": {
            "llama_guard_8b": True,
            "indobert_toxic": False
        },
        "thresholds": {
            "indobert_threshold": 0.75
        }
    }
)
```

### 6.3 Metrics and Analytics

```python
# Get system metrics
metrics = requests.get(
    "http://localhost:8000/metrics",
    headers=headers
).json()

print(f"Total prompts: {metrics['total_prompts']}")
print(f"Block rate: {metrics['block_percentage']}%")
print(f"Average latency: {metrics['avg_latency']}ms")
```

## ⚙️ Step 7: Advanced Configuration

### 7.1 Model Optimization

**Memory Optimization (Recommended for production):**
```bash
# Enable 4-bit quantization for 75% memory reduction
USE_QUANTIZATION=true
QUANTIZATION_BITS=4
QUANTIZATION_TYPE=nf4
```

**Performance Optimization:**
```bash
# Use CUDA if available
DEVICE=cuda

# Optimize data types
COMPUTE_DTYPE=float16
TORCH_DTYPE=float16
```

### 7.2 Production Deployment with Gunicorn

```bash
# Install Gunicorn
pip install gunicorn[uvicorn]

# Start with production settings
gunicorn main:app -c gunicorn.conf.py
```

### 7.3 Load Balancing with Nginx

```nginx
upstream guard_api {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://guard_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 120s;
    }
}
```

## 🛡️ Step 8: Security Best Practices

### 8.1 API Key Management

```bash
# Generate secure API keys
openssl rand -hex 32

# Set environment variables
export API_KEY="your-secure-production-key"
export REACT_APP_API_KEY="your-secure-production-key"
```

### 8.2 CORS Configuration

```bash
# Restrict CORS origins in production
CORS_ORIGINS=["https://your-domain.com"]
```

### 8.3 Rate Limiting

The system includes built-in rate limiting. Configure in `config.py`:

```python
RATE_LIMIT_REQUESTS = 100  # requests per minute
RATE_LIMIT_WINDOW = 60     # window in seconds
```

## 📊 Step 9: Monitoring and Troubleshooting

### 9.1 Health Monitoring

```bash
# Check system health
curl http://localhost:8000/healthz

# Monitor logs
docker logs -f <container_name>

# Check Redis connectivity
redis-cli ping
```

### 9.2 Common Issues and Solutions

**Issue: ModuleNotFoundError for advanced_metrics**
```bash
# Solution: Rebuild Docker image
docker build -f Dockerfile.production -t tupaikapitalis/guard-api:latest .
```

**Issue: CUDA out of memory**
```bash
# Solution: Enable quantization
USE_QUANTIZATION=true
QUANTIZATION_BITS=4
```

**Issue: Model loading fails**
```bash
# Solution: Check Hugging Face token and model access
export HUGGINGFACE_TOKEN="your-valid-token"
```

### 9.3 Performance Monitoring

```python
# Monitor key metrics
import requests

metrics = requests.get(
    "http://localhost:8000/analytics/dashboard",
    headers={"Authorization": "Bearer your-api-key"}
).json()

print(f"Response time: {metrics['avg_response_time']}ms")
print(f"Cache hit rate: {metrics['cache_hit_rate']}%")
print(f"Guard accuracy: {metrics['guard_accuracy']}%")
```

## 🚀 Step 10: Production Deployment

### 10.1 Environment Preparation

```bash
# Set production environment variables
export API_KEY="your-secure-production-key"
export DEVICE="cuda"  # or "cpu"
export USE_QUANTIZATION="true"
export REDIS_URL="redis://your-redis-server:6379"
```

### 10.2 Deploy to Cloud

**Using Docker Hub image:**
```bash
# Pull and run the latest image
docker pull tupaikapitalis/guard-api:latest
docker run -d \
  --name guard-api \
  -p 8000:8000 \
  -e API_KEY="your-production-key" \
  tupaikapitalis/guard-api:latest
```

**Using Docker Compose:**
```bash
# Deploy with production configuration
docker-compose -f docker-compose.production.yml up -d
```

### 10.3 Systemd Service (Linux)

Create `/etc/systemd/system/guard-api.service`:
```ini
[Unit]
Description=Guard API Service
After=network.target

[Service]
Type=exec
User=www-data
WorkingDirectory=/path/to/GUARD-SIMPLE
Environment=API_KEY=your-production-key
ExecStart=/usr/local/bin/gunicorn main:app -c gunicorn.conf.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable guard-api
sudo systemctl start guard-api
```

## 📈 Understanding the Guards

### Guard Types and Capabilities

1. **LLaMA Guard 8B/1B**: Advanced safety classification
   - Detects: Violence, sexual content, hate speech, cybercrime, PII
   - Memory: 4-16GB (with quantization)
   - Accuracy: Very High

2. **IndoBERT Toxic Classifier**: Indonesian language toxicity
   - Detects: Toxic language, hate speech
   - Memory: <1GB
   - Languages: Indonesian, English

3. **LLM Guard**: Multi-validator scanner
   - Detects: Prompt injection, PII, URLs, sensitive data
   - Memory: <2GB
   - Speed: Very Fast

4. **NeMo Guardrails**: Policy-based filtering
   - Detects: Policy violations, content categories
   - Memory: <500MB
   - Customizable: Yes

### Decision Logic

- **BLOCK**: If any guard returns "block"
- **WARN**: If any guard returns "warn" (and none block)
- **ALLOW**: If all guards return "allow"

## 🔧 Customization and Extension

### Adding Custom Guards

1. Create a new guard class:
```python
class CustomGuard(BaseGuard):
    def __init__(self):
        super().__init__("custom_guard")
    
    async def analyze(self, prompt: str, **kwargs):
        # Your custom logic here
        return GuardResult(
            verdict="allow",  # or "warn" or "block"
            labels=["custom_category"],
            score=0.5
        )
```

2. Register in `GuardManager`:
```python
self.guards["custom_guard"] = CustomGuard()
```

### Custom Thresholds

Adjust sensitivity in the Settings tab or via API:
```python
requests.put(
    "http://localhost:8000/config",
    headers=headers,
    json={
        "thresholds": {
            "indobert_threshold": 0.8,  # Higher = more strict
            "custom_threshold": 0.6
        }
    }
)
```

## 📚 API Reference

### Core Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/healthz` | GET | Health check | No |
| `/analyze` | POST | Analyze prompt | Yes |
| `/config` | GET/PUT | Configuration | Yes |
| `/metrics` | GET | System metrics | Yes |
| `/analytics/*` | GET | Advanced analytics | Yes |

### Response Formats

**Analyze Response:**
```json
{
  "final": "allow|warn|block",
  "per_guard": {
    "guard_name": {
      "verdict": "allow|warn|block",
      "labels": ["category1", "category2"],
      "score": 0.85,
      "latency_ms": 150
    }
  },
  "policy": {
    "rule": "block if any block; warn if any warn; else allow"
  },
  "metadata": {
    "total_latency_ms": 200,
    "cache_hit": false,
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

## 🎓 Next Steps

Congratulations! You've successfully deployed the GUARD-SIMPLE API. Here are some next steps:

1. **Integrate with your application**: Use the API endpoints to screen prompts
2. **Monitor performance**: Use the dashboard to track metrics and optimize
3. **Customize guards**: Add your own safety rules and thresholds
4. **Scale deployment**: Use load balancing and multiple instances
5. **Enhance security**: Implement additional authentication and monitoring

## 🆘 Support and Troubleshooting

### Getting Help

1. **Check logs**: `docker logs <container_name>`
2. **API documentation**: http://localhost:8000/docs
3. **Health status**: http://localhost:8000/healthz
4. **Test authentication**: Use the provided curl examples

### Common Solutions

- **Memory issues**: Enable quantization with `USE_QUANTIZATION=true`
- **Slow performance**: Use GPU with `DEVICE=cuda`
- **Connection errors**: Check Redis server status
- **Authentication errors**: Verify API key matches in both backend and frontend

### Performance Optimization

- **Memory**: Use 4-bit quantization for 75% reduction
- **Speed**: Enable GPU acceleration when available
- **Caching**: Redis caching reduces duplicate analysis
- **Load balancing**: Use multiple workers for high traffic

## 📄 License

MIT License - see LICENSE file for details.

---

**🎉 You're all set!** Your GUARD-SIMPLE API is now ready to protect your LLM applications from unsafe prompts. Visit the dashboard at http://localhost:3000 to start testing and monitoring your deployment.