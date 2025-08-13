from fastapi import FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
import asyncio
import time
import logging
import hashlib
import json
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis
import os
from guards import GuardManager
from config import Config
from metrics import MetricsCollector
from advanced_metrics import AdvancedMetricsCollector, TimeWindow

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()
API_KEY = os.getenv("API_KEY", "guard-api-key-2024")

app = FastAPI(title="Prompt Railguarding API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
guard_manager = GuardManager()
config = Config()
metrics = MetricsCollector()
advanced_metrics = AdvancedMetricsCollector(metrics)

# Authentication dependency
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the API key from Bearer token"""
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Pydantic models
class AnalyzeRequest(BaseModel):
    prompt: str
    user_id: Optional[str] = None
    lang: str = "auto"

class GuardResult(BaseModel):
    verdict: str
    labels: List[str]
    score: Optional[float] = None

class AnalyzeResponse(BaseModel):
    final: str
    per_guard: Dict[str, Dict[str, Any]]
    policy: Dict[str, str]

class ConfigUpdate(BaseModel):
    enable_all: Optional[bool] = None
    guards: Optional[Dict[str, bool]] = None
    thresholds: Optional[Dict[str, float]] = None

# In startup event
@app.on_event("startup")
async def startup():
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = redis.from_url(redis_url)
    await FastAPILimiter.init(redis_client)

# Apply to endpoints
@app.post("/analyze", dependencies=[Depends(RateLimiter(times=50, seconds=60)), Depends(verify_api_key)])
async def analyze_prompt(request: AnalyzeRequest):
    """Analyze a prompt through all enabled guards with Redis caching"""
    start_time = time.time()
    
    try:
        # Hash prompt for logging (privacy)
        prompt_hash = hashlib.sha256(request.prompt.encode()).hexdigest()[:16]
        logger.info(f"Analyzing prompt {prompt_hash} for user {request.user_id}")
        
        # Generate cache key based on prompt, language, enabled guards, and thresholds
        enabled_guards = config.get_enabled_guards()
        thresholds = config.get_thresholds()
        cache_key_data = {
            "prompt": request.prompt,
            "lang": request.lang,
            "guards": sorted(enabled_guards),
            "thresholds": sorted(thresholds.items())
        }
        cache_key = f"analysis:{hashlib.sha256(json.dumps(cache_key_data, sort_keys=True).encode()).hexdigest()}"
        
        # Try to get cached result
        redis_client = None
        cached_result = None
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            redis_client = redis.from_url(redis_url)
            cached_data = await redis_client.get(cache_key)
            if cached_data:
                cached_result = json.loads(cached_data)
                metrics.record_cache_hit()
                logger.info(f"Cache hit for prompt {prompt_hash}")
                
                # Return cached result
                response = AnalyzeResponse(
                    final=cached_result["final"],
                    per_guard=cached_result["per_guard"],
                    policy=cached_result["policy"]
                )
                
                # Record metrics for cached response
                latency = time.time() - start_time
                metrics.record_analysis(cached_result["final"], cached_result["per_guard"], latency)
                
                # Record advanced metrics for cached response
                advanced_metrics.record_advanced_analysis(
                    prompt=request.prompt,
                    user_id=request.user_id,
                    final_verdict=cached_result["final"],
                    guard_results=cached_result["per_guard"],
                    latency=latency,
                    request_time=start_time
                )
                
                logger.info(f"Analysis complete (cached): {cached_result['final']} in {latency:.3f}s")
                return response
            else:
                metrics.record_cache_miss()
                logger.info(f"Cache miss for prompt {prompt_hash}")
        except Exception as cache_error:
            logger.warning(f"Cache lookup failed: {cache_error}")
            metrics.record_cache_miss()
        
        # Run guards in parallel (cache miss or cache error)
        guard_results = await guard_manager.analyze_prompt(
            request.prompt, 
            request.lang,
            enabled_guards,
            thresholds
        )
        
        # Apply policy decision
        final_verdict = apply_policy(guard_results)
        
        # Prepare response
        response_data = {
            "final": final_verdict,
            "per_guard": guard_results,
            "policy": {"rule": "block if any block; warn if any warn; else allow"}
        }
        
        # Cache the result (expire after 1 hour)
        try:
            if redis_client:
                await redis_client.setex(cache_key, 3600, json.dumps(response_data))
                logger.info(f"Cached result for prompt {prompt_hash}")
        except Exception as cache_error:
            logger.warning(f"Failed to cache result: {cache_error}")
        finally:
            if redis_client:
                await redis_client.close()
        
        # Update metrics
        latency = time.time() - start_time
        metrics.record_analysis(final_verdict, guard_results, latency)
        
        # Record advanced metrics
        advanced_metrics.record_advanced_analysis(
            prompt=request.prompt,
            user_id=request.user_id,
            final_verdict=final_verdict,
            guard_results=guard_results,
            latency=latency,
            request_time=start_time
        )
        
        response = AnalyzeResponse(
            final=final_verdict,
            per_guard=guard_results,
            policy={"rule": "block if any block; warn if any warn; else allow"}
        )
        
        logger.info(f"Analysis complete: {final_verdict} in {latency:.3f}s")
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        # Fail-safe: return warn on error
        return AnalyzeResponse(
            final="warn",
            per_guard={"error": {"verdict": "warn", "labels": ["system_error"], "score": None}},
            policy={"rule": "fail-safe on error"}
        )

@app.get("/metrics", dependencies=[Depends(verify_api_key)])
async def get_metrics():
    """Get system metrics and statistics"""
    return metrics.get_all_metrics()

@app.get("/analytics/dashboard", dependencies=[Depends(verify_api_key)])
async def get_analytics_dashboard():
    """Get comprehensive analytics dashboard data"""
    return advanced_metrics.get_comprehensive_dashboard()

@app.get("/analytics/time/{window}", dependencies=[Depends(verify_api_key)])
async def get_time_analytics(window: str):
    """Get time-based analytics for specified window (1h, 24h, 7d, 30d)"""
    try:
        time_window = TimeWindow(window)
        return advanced_metrics.get_time_based_analytics(time_window)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid time window. Use: 1h, 24h, 7d, or 30d")

@app.get("/analytics/users", dependencies=[Depends(verify_api_key)])
async def get_user_analytics(limit: int = 20):
    """Get user behavior analytics"""
    return advanced_metrics.get_user_behavior_analytics(limit)

@app.get("/analytics/threats", dependencies=[Depends(verify_api_key)])
async def get_threat_intelligence():
    """Get threat intelligence and security patterns"""
    return advanced_metrics.get_threat_intelligence()

@app.get("/analytics/performance", dependencies=[Depends(verify_api_key)])
async def get_model_performance():
    """Get model performance trends and health metrics"""
    return advanced_metrics.get_model_performance_trends()

@app.get("/analytics/anomalies", dependencies=[Depends(verify_api_key)])
async def get_anomalies():
    """Get recent system anomalies"""
    return {"anomalies": advanced_metrics.get_anomalies()}

@app.get("/config", dependencies=[Depends(verify_api_key)])
async def get_config():
    """Get current configuration"""
    return {
        "enable_all": config.enable_all,
        "guards": config.guards,
        "thresholds": config.thresholds,
        "guard_versions": guard_manager.get_versions()
    }

@app.put("/config", dependencies=[Depends(verify_api_key)])
async def update_config(update: ConfigUpdate):
    """Update configuration"""
    try:
        if update.enable_all is not None:
            config.enable_all = update.enable_all
            
        if update.guards:
            config.guards.update(update.guards)
            
        if update.thresholds:
            config.thresholds.update(update.thresholds)
            
        logger.info("Configuration updated")
        return {"status": "success", "message": "Configuration updated"}
        
    except Exception as e:
        logger.error(f"Config update failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/cache", dependencies=[Depends(verify_api_key)])
async def clear_cache():
    """Clear the Redis cache"""
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = redis.from_url(redis_url)
        await redis_client.flushdb()  # Clear current database
        await redis_client.close()
        
        # Reset cache metrics
        metrics.cache_hits = 0
        metrics.cache_misses = 0
        
        logger.info("Cache cleared successfully")
        return {"status": "success", "message": "Cache cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Prompt Railguarding API",
        "version": "1.0.0",
        "status": "running",
        "health_endpoint": "/healthz",
        "docs": "/docs"
    }

@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "guards": guard_manager.health_check(),
        "redis": await check_redis_health()
    }
    
    # Determine overall health
    if not health_status["redis"] or not all(health_status["guards"].values()):
        health_status["status"] = "degraded"
    
    return health_status

async def check_redis_health() -> bool:
    """Check Redis connectivity"""
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = redis.from_url(redis_url)
        await redis_client.ping()
        await redis_client.close()
        return True
    except:
        return False

def apply_policy(guard_results: Dict[str, Dict]) -> str:
    """Apply the guarding policy to determine final verdict"""
    verdicts = [result["verdict"] for result in guard_results.values()]
    
    if "block" in verdicts:
        return "block"
    elif "warn" in verdicts:
        return "warn"
    else:
        return "allow"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)