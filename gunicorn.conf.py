# Gunicorn configuration file for Guard API
# Production-ready settings for FastAPI deployment

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
preload_app = True

# Timeout settings (important for AI model loading)
timeout = 120
keepalive = 5
graceful_timeout = 30

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "guard-api"

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# SSL (uncomment and configure for HTTPS)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

# Performance tuning
# worker_tmp_dir = "/dev/shm"  # Use RAM for worker temp files (Linux only)
# On macOS, use default temp directory or specify: worker_tmp_dir = "/tmp"
worker_tmp_dir = "/tmp"  # Cross-platform compatible temp directory

# Restart workers after this many requests to prevent memory leaks
max_requests = 1000
max_requests_jitter = 100

# Restart workers if they haven't processed a request in this many seconds
worker_timeout = 120

# Environment variables (optional - can also be set in shell)
# raw_env = [
#     "API_KEY=your-production-api-key",
#     "REDIS_URL=redis://localhost:6379",
#     "DEVICE=cuda"
# ]