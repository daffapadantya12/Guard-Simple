import time
from typing import Dict, List, Any
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and manages system metrics"""
    
    def __init__(self):
        # Decision counters
        self.total_requests = 0
        self.decision_counts = {
            "allow": 0,
            "warn": 0,
            "block": 0
        }
        
        # Per-guard statistics
        self.guard_stats = defaultdict(lambda: {
            "total": 0,
            "allow": 0,
            "warn": 0,
            "block": 0,
            "errors": 0,
            "total_latency": 0.0,
            "avg_latency": 0.0
        })
        
        # Time series data (last 100 requests)
        self.decision_history = deque(maxlen=100)
        self.latency_history = deque(maxlen=100)
        
        # Cache metrics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Category tracking
        self.category_counts = defaultdict(int)
        
        # Error tracking
        self.error_counts = defaultdict(int)
        
        # Start time for uptime calculation
        self.start_time = time.time()
        
        logger.info("Metrics collector initialized")
    
    def record_analysis(self, final_verdict: str, guard_results: Dict, latency: float):
        """Record metrics for a completed analysis"""
        self.total_requests += 1
        self.decision_counts[final_verdict] += 1
        
        # Record time series
        timestamp = time.time()
        self.decision_history.append({
            "timestamp": timestamp,
            "verdict": final_verdict
        })
        self.latency_history.append({
            "timestamp": timestamp,
            "latency": latency
        })
        
        # Record per-guard stats
        for guard_name, result in guard_results.items():
            stats = self.guard_stats[guard_name]
            stats["total"] += 1
            
            verdict = result.get("verdict", "error")
            if verdict in ["allow", "warn", "block"]:
                stats[verdict] += 1
            else:
                stats["errors"] += 1
                self.error_counts[guard_name] += 1
            
            # Update latency (mock per-guard latency)
            guard_latency = latency / len(guard_results)  # Simplified
            stats["total_latency"] += guard_latency
            stats["avg_latency"] = stats["total_latency"] / stats["total"]
            
            # Record categories
            labels = result.get("labels", [])
            for label in labels:
                self.category_counts[label] += 1
        
        logger.debug(f"Recorded metrics for {final_verdict} decision with {latency:.3f}s latency")
    
    def record_cache_hit(self):
        """Record a cache hit"""
        self.cache_hits += 1
    
    def record_cache_miss(self):
        """Record a cache miss"""
        self.cache_misses += 1
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate percentages
        total = max(self.total_requests, 1)  # Avoid division by zero
        percentages = {
            "allow_pct": (self.decision_counts["allow"] / total) * 100,
            "warn_pct": (self.decision_counts["warn"] / total) * 100,
            "block_pct": (self.decision_counts["block"] / total) * 100
        }
        
        # Calculate average latency
        avg_latency = 0.0
        if self.latency_history:
            avg_latency = sum(item["latency"] for item in self.latency_history) / len(self.latency_history)
        
        # Cache hit rate
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = 0.0
        if total_cache_requests > 0:
            cache_hit_rate = (self.cache_hits / total_cache_requests) * 100
        
        # Recent activity (last 10 minutes)
        recent_cutoff = current_time - 600  # 10 minutes
        recent_decisions = [
            item for item in self.decision_history 
            if item["timestamp"] > recent_cutoff
        ]
        
        return {
            "overview": {
                "total_requests": self.total_requests,
                "uptime_seconds": uptime,
                "avg_latency_ms": avg_latency * 1000,
                "requests_per_minute": (self.total_requests / (uptime / 60)) if uptime > 0 else 0
            },
            "decisions": {
                "counts": self.decision_counts.copy(),
                "percentages": percentages,
                "recent_activity": len(recent_decisions)
            },
            "per_guard_stats": dict(self.guard_stats),
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate_pct": cache_hit_rate
            },
            "categories": {
                "frequent_triggers": dict(sorted(
                    self.category_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10])  # Top 10 categories
            },
            "errors": {
                "by_guard": dict(self.error_counts),
                "total_errors": sum(self.error_counts.values())
            },
            "time_series": {
                "recent_decisions": list(self.decision_history)[-20:],  # Last 20
                "recent_latencies": list(self.latency_history)[-20:]   # Last 20
            }
        }
    
    def get_kpi_summary(self) -> Dict[str, Any]:
        """Get key performance indicators for dashboard"""
        total = max(self.total_requests, 1)
        
        # Calculate average latency
        avg_latency = 0.0
        if self.latency_history:
            avg_latency = sum(item["latency"] for item in self.latency_history) / len(self.latency_history)
        
        return {
            "total_prompts": self.total_requests,
            "block_percentage": (self.decision_counts["block"] / total) * 100,
            "warn_percentage": (self.decision_counts["warn"] / total) * 100,
            "avg_latency_ms": avg_latency * 1000,
            "uptime_hours": (time.time() - self.start_time) / 3600
        }
    
    def get_guard_performance(self) -> List[Dict[str, Any]]:
        """Get per-guard performance table data"""
        performance_data = []
        
        for guard_name, stats in self.guard_stats.items():
            total = max(stats["total"], 1)
            error_rate = (stats["errors"] / total) * 100
            block_rate = (stats["block"] / total) * 100
            warn_rate = (stats["warn"] / total) * 100
            
            performance_data.append({
                "guard_name": guard_name,
                "total_requests": stats["total"],
                "avg_latency_ms": stats["avg_latency"] * 1000,
                "error_rate_pct": error_rate,
                "block_rate_pct": block_rate,
                "warn_rate_pct": warn_rate,
                "status": "healthy" if error_rate < 5 else "degraded"
            })
        
        return sorted(performance_data, key=lambda x: x["total_requests"], reverse=True)
    
    def get_decision_chart_data(self) -> Dict[str, List]:
        """Get data for decision counts chart"""
        # Group decisions by time buckets (e.g., last 24 hours in hourly buckets)
        current_time = time.time()
        hourly_buckets = {}
        
        # Initialize 24 hourly buckets
        for i in range(24):
            bucket_time = current_time - (i * 3600)  # i hours ago
            bucket_key = int(bucket_time // 3600) * 3600  # Round to hour
            hourly_buckets[bucket_key] = {"allow": 0, "warn": 0, "block": 0}
        
        # Fill buckets with actual data
        for decision in self.decision_history:
            bucket_key = int(decision["timestamp"] // 3600) * 3600
            if bucket_key in hourly_buckets:
                hourly_buckets[bucket_key][decision["verdict"]] += 1
        
        # Convert to chart format
        timestamps = sorted(hourly_buckets.keys())
        return {
            "timestamps": timestamps,
            "allow": [hourly_buckets[ts]["allow"] for ts in timestamps],
            "warn": [hourly_buckets[ts]["warn"] for ts in timestamps],
            "block": [hourly_buckets[ts]["block"] for ts in timestamps]
        }
    
    def reset_metrics(self):
        """Reset all metrics (for testing/debugging)"""
        self.__init__()
        logger.info("All metrics reset")