# Prompt Railguarding API - MVP

A comprehensive API system that screens user prompts before sending them to an LLM. Uses multiple AI safety models including LLaMA Guard 8B, LLaMA Guard 1B, IndoBERT Toxic Classifier, LLM Guard, and NeMo Guardrails to detect and block unsafe, malicious, or policy-violating input.

## Features

### Backend (FastAPI)
- **Multiple Guard Integration**: LLaMA Guard 8B/1B, IndoBERT, LLM Guard, NeMo Guardrails
- **Parallel Processing**: All guards run concurrently for optimal performance
- **Rate Limiting**: Configurable request limits per IP
- **Caching**: Redis-based caching for duplicate prompt detection
- **Comprehensive Metrics**: Real-time statistics and performance monitoring
- **Health Monitoring**: System health checks and guard status monitoring
- **Configurable Thresholds**: Adjustable sensitivity settings per guard

### Frontend (React)
- **Input & Analysis Tab**: Interactive prompt testing with detailed results
- **Dashboard Tab**: Real-time metrics, KPIs, and performance charts
- **Settings Tab**: Guard toggles, threshold adjustments, and configuration
- **Responsive Design**: Modern UI with Tailwind CSS
- **Real-time Updates**: Live metrics and health status monitoring

## Tech Stack

- **Backend**: FastAPI (Python 3.11+), Redis, uvicorn/gunicorn
- **Frontend**: React 18, Tailwind CSS, Axios, Recharts
- **AI Models**: LLaMA Guard, IndoBERT, LLM Guard, NeMo Guardrails
- **Caching**: Redis with fastapi-cache2
- **Rate Limiting**: fastapi-limiter

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 16+
- Redis server

### Backend Setup

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Configure Environment Variables**:

Run the interactive setup script:
```bash
python setup_env.py
```

Or manually create a `.env` file:
```bash
# Hugging Face Configuration
HUGGINGFACE_TOKEN=your_token_here

# Device Configuration (auto/cuda/cpu)
DEVICE=auto

# Model Quantization (reduces memory usage)
USE_QUANTIZATION=true
QUANTIZATION_BITS=4
QUANTIZATION_TYPE=nf4
USE_DOUBLE_QUANT=true
COMPUTE_DTYPE=float16

# Model Loading Configuration
TRUST_REMOTE_CODE=true
TORCH_DTYPE=float16
HF_CACHE_DIR=./models_cache
```

**Important**: Get your Hugging Face token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and request access to LLaMA Guard models.

3. **Start Redis server**:
```bash
# macOS with Homebrew
brew services start redis

# Or run directly
redis-server
```

3. **Start the FastAPI server**:
```bash
# Development
python main.py
# or
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend Setup

1. **Install Node.js dependencies**:
```bash
npm install
```

2. **Start the React development server**:
```bash
npm start
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Model Configuration

### Environment Variables

The system supports extensive configuration through environment variables:

| Variable | Options | Description |
|----------|---------|-------------|
| `HUGGINGFACE_TOKEN` | string | Your HF token for accessing gated models |
| `DEVICE` | `auto`, `cuda`, `cpu` | Compute device selection |
| `USE_QUANTIZATION` | `true`, `false` | Enable model quantization |
| `QUANTIZATION_BITS` | `4`, `8` | Quantization precision |
| `QUANTIZATION_TYPE` | `nf4`, `fp4` | Quantization algorithm |
| `USE_DOUBLE_QUANT` | `true`, `false` | Double quantization for better compression |
| `COMPUTE_DTYPE` | `float16`, `float32` | Computation data type |
| `TORCH_DTYPE` | `float16`, `float32` | Model weights data type |
| `HF_CACHE_DIR` | path | Directory for cached models |

### Quantization Benefits

**4-bit Quantization** (Recommended):
- 🔥 **75% memory reduction** (16GB → 4GB for 7B models)
- ⚡ **2-3x faster inference** on compatible hardware
- 📊 **Minimal quality loss** (<2% accuracy drop)
- 💰 **Cost effective** for production deployment

**8-bit Quantization**:
- 🔥 **50% memory reduction** (16GB → 8GB for 7B models)
- ⚡ **1.5-2x faster inference**
- 📊 **Higher quality** (<1% accuracy drop)
- ⚖️ **Balanced** memory/quality trade-off

### Model Loading Behavior

1. **LlamaGuard8B**: Attempts to load `meta-llama/LlamaGuard-7b`
   - Requires HF authentication and model access approval
   - Falls back to rule-based classification if loading fails
   - Supports quantization for memory efficiency

2. **LlamaGuard1B**: Uses `unitary/toxic-bert` as alternative
   - Publicly available, no authentication required
   - Smaller model, faster inference
   - Good baseline safety classification

3. **IndoBERTToxic**: Indonesian language toxicity detection
   - Specialized for Indonesian content
   - Uses `unitary/toxic-bert` with language detection

### Hardware Requirements

| Configuration | GPU Memory | CPU RAM | Performance |
|---------------|------------|---------|-------------|
| **No Quantization** | 16GB+ | 32GB+ | Highest quality |
| **8-bit Quantization** | 8GB+ | 16GB+ | Balanced |
| **4-bit Quantization** | 4GB+ | 8GB+ | Most efficient |
| **CPU Only** | N/A | 16GB+ | Slower but works |

## API Endpoints

### Authentication
All API endpoints (except `/healthz`) require API key authentication using Bearer token:

```bash
# Set your API key
export API_KEY="your_api_key_here"

# Include in requests
curl -H "Authorization: Bearer $API_KEY" http://localhost:8000/config
```

### Public Endpoints

#### GET /healthz
Health check endpoint (no authentication required).

### Protected Endpoints

#### POST /analyze
Analyze a prompt through all enabled guards. **Requires API key authentication.**

**Request**:
```json
{
  "prompt": "Your prompt text here",
  "user_id": "optional-user-id",
  "lang": "auto|en|id"
}
```

**Response**:
```json
{
  "final": "allow|warn|block",
  "per_guard": {
    "llama_guard_8b": {
      "verdict": "block",
      "labels": ["Violence"],
      "score": null
    },
    "indobert_toxic": {
      "verdict": "warn",
      "labels": ["toxicity"],
      "score": 0.82
    }
  },
  "policy": {
    "rule": "block if any block; warn if any warn; else allow"
  }
}
```

#### GET /metrics
Retrieve comprehensive system metrics and statistics. **Requires API key authentication.**

#### GET /config
Get current configuration including guard toggles and thresholds. **Requires API key authentication.**

#### PUT /config
Update system configuration. **Requires API key authentication.**

**Request**:
```json
{
  "enable_all": true,
  "guards": {
    "llama_guard_8b": true,
    "indobert_toxic": false
  },
  "thresholds": {
    "indobert_threshold": 0.75
  }
}
```

### GET /healthz
Health check endpoint for system monitoring.

## Guard Configuration

### Decision Policy
- **BLOCK**: If any enabled guard returns "block"
- **WARN**: If any enabled guard returns "warn" (and none return "block")
- **ALLOW**: If all enabled guards return "allow"

### Guard Details

1. **LLaMA Guard 8B/1B**: Advanced safety classification
   - Categories: Violence, sexual content, hate speech, cybercrime, PII
   - Disagreement resolution: Choose more restrictive verdict

2. **IndoBERT Toxic Classifier**: Indonesian toxicity detection
   - Score ≥ 0.95: Block
   - Score ≥ threshold (default 0.70): Warn
   - Configurable threshold via settings

3. **LLM Guard**: Multi-validator prompt scanner
   - Validators: PromptInjection, Toxicity, PII, URL, SensitiveData
   - Critical severity: Block
   - Moderate severity: Warn

4. **NeMo Guardrails**: Policy-based content filtering
   - Block: Violence, minors, cybercrime, privacy
   - Warn: Hate speech, self-harm

## Monitoring & Metrics

### Key Performance Indicators
- Total prompts processed
- Block/warn/allow percentages
- Average response latency
- Cache hit rate
- Per-guard performance statistics

### Health Monitoring
- Guard availability and error rates
- Redis connectivity
- System uptime and performance
- Real-time status indicators

## Security Features

- **API Key Authentication**: Bearer token authentication for all protected endpoints
- **Privacy Protection**: Prompts are hashed in logs, only redacted snippets stored
- **Rate Limiting**: Configurable per-IP request limits
- **CORS Protection**: Restricted to authorized origins
- **Fail-Safe Design**: Errors default to "warn" verdict
- **Input Validation**: Comprehensive request validation

## Development

### Project Structure
```
├── main.py              # FastAPI application
├── guards.py            # Guard implementations
├── config.py            # Configuration management
├── metrics.py           # Metrics collection
├── requirements.txt     # Python dependencies
├── package.json         # Node.js dependencies
├── src/
│   ├── App.js          # Main React application
│   ├── App.css         # Styles and animations
│   └── index.js        # React entry point
└── public/
    └── index.html      # HTML template
```

### Adding New Guards

1. Create a new guard class inheriting from `BaseGuard`
2. Implement the `analyze()` method
3. Add the guard to `GuardManager`
4. Update configuration and UI as needed

### Customizing Thresholds

Thresholds can be adjusted via:
- API: PUT /config endpoint
- UI: Settings tab sliders
- Code: Update `config.py` defaults

## Testing

Run the test suite:
```bash
python test_advanced_analytics.py
python test_api_auth.py
python test_rate_limit.py
```

## Production Deployment with Gunicorn

### Prerequisites

Install Gunicorn:
```bash
pip install gunicorn[uvicorn]
```

### Basic Deployment

For development/testing:
```bash
gunicorn main:app --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Production Deployment

Use the provided configuration file for production:
```bash
gunicorn main:app -c gunicorn.conf.py
```

### Configuration Options

The `gunicorn.conf.py` file includes production-ready settings:
- **Workers**: 4 worker processes for handling concurrent requests
- **Worker Class**: `uvicorn.workers.UvicornWorker` for FastAPI compatibility
- **Timeout**: 120 seconds to accommodate AI model loading
- **Logging**: Structured access and error logging
- **Security**: Request size limits and security headers
- **Performance**: Memory optimization and worker recycling

### Custom Configuration

You can override settings via command line:
```bash
# Custom number of workers
gunicorn main:app -c gunicorn.conf.py --workers 8

# Custom bind address
gunicorn main:app -c gunicorn.conf.py --bind 127.0.0.1:8080

# Enable SSL
gunicorn main:app -c gunicorn.conf.py --keyfile /path/to/key.pem --certfile /path/to/cert.pem
```

### Environment Variables for Production

Set these environment variables before starting:
```bash
export API_KEY="your-secure-production-api-key"
export REDIS_URL="redis://localhost:6379"
export DEVICE="cuda"  # or "cpu"
export HF_TOKEN="your-huggingface-token"
```

### Docker Deployment

The included Dockerfile uses Gunicorn by default. Build and run:
```bash
# Build the image
docker build -t guard-api .

# Run with environment variables
docker run -d \
  --name guard-api \
  -p 8000:8000 \
  -e API_KEY="your-production-api-key" \
  -e DEVICE="cuda" \
  guard-api
```

Or use docker-compose:
```bash
docker-compose up -d
```

### Systemd Service (Linux)

Create a systemd service for automatic startup:

1. Create service file `/etc/systemd/system/guard-api.service`:
```ini
[Unit]
Description=Guard API Service
After=network.target

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory=/path/to/GUARD-SIMPLE
Environment=PATH=/path/to/venv/bin
Environment=API_KEY=your-production-api-key
Environment=DEVICE=cuda
ExecStart=/path/to/venv/bin/gunicorn main:app -c gunicorn.conf.py
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

2. Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable guard-api
sudo systemctl start guard-api
sudo systemctl status guard-api
```

### Performance Considerations

- **Workers**: Set to `2 * CPU_CORES + 1` for CPU-bound tasks
- **Memory**: Each worker loads the AI models (~4-8GB per worker)
- **GPU**: Use `DEVICE=cuda` only if GPU memory can handle multiple workers
- **Timeout**: Increase for large model loading or complex analysis

### Health Checks

Gunicorn provides built-in health monitoring:
```bash
# Check if workers are responding
curl http://localhost:8000/healthz

# Monitor worker status
ps aux | grep gunicorn

# View logs
tail -f /var/log/guard-api/access.log
tail -f /var/log/guard-api/error.log
```

### Load Balancing

For high-traffic deployments, use a reverse proxy:

**Nginx configuration example:**
```nginx
upstream guard_api {
    server 127.0.0.1:8000;
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
}

server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://guard_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }
}
```

### Backend
```bash
# Using gunicorn with multiple workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Frontend
```bash
# Build for production
npm run build

# Serve with nginx or similar
```

### Environment Variables
- `API_KEY`: API authentication key for backend (default: `guard-api-key-2024`)
- `REACT_APP_API_KEY`: API authentication key for frontend (should match API_KEY)
- `REDIS_URL`: Redis connection string
- `CORS_ORIGINS`: Allowed CORS origins
- `LOG_LEVEL`: Logging level (INFO, DEBUG, etc.)

### API Authentication

The API now requires Bearer token authentication for all endpoints except:
- `/docs` - API documentation
- `/healthz` - Health check endpoint
- `/` - Root endpoint

**Setting up API Key:**
1. Set `API_KEY` in your `.env` file for the backend
2. Set `REACT_APP_API_KEY` in your `.env` file for the frontend (must match backend key)
3. For external API calls, include the header: `Authorization: Bearer YOUR_API_KEY`

**Example API Request:**
```bash
curl -H "Authorization: Bearer your-api-key-here" \
     -H "Content-Type: application/json" \
     http://localhost:8000/config
```

**Testing Authentication:**
```bash
# Test the authentication system
python test_auth.py
```

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review logs for error details
3. Monitor health status at `/healthz`
4. Check Redis connectivity and guard status