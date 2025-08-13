import time
import json
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics
import logging
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

class TimeWindow(Enum):
    HOUR = "1h"
    DAY = "24h"
    WEEK = "7d"
    MONTH = "30d"

@dataclass
class ThreatIntelligence:
    pattern: str
    severity: str
    frequency: int
    first_seen: float
    last_seen: float
    affected_guards: List[str]

class AdvancedMetricsCollector:
    """Enhanced metrics collector with advanced analytics capabilities"""
    
    def __init__(self, base_metrics):
        self.base_metrics = base_metrics
        
        # Time-based analytics
        self.hourly_stats = defaultdict(lambda: {
            "requests": 0, "blocks": 0, "warns": 0, "allows": 0,
            "avg_latency": 0.0, "total_latency": 0.0
        })
        self.daily_stats = defaultdict(lambda: {
            "requests": 0, "blocks": 0, "warns": 0, "allows": 0,
            "unique_users": set(), "peak_hour": 0, "peak_requests": 0
        })
        
        # User behavior analytics
        self.user_patterns = defaultdict(lambda: {
            "total_requests": 0,
            "block_rate": 0.0,
            "warn_rate": 0.0,
            "avg_prompt_length": 0.0,
            "common_categories": defaultdict(int),
            "request_times": deque(maxlen=100),
            "risk_score": 0.0
        })
        
        # Model performance trends
        self.model_trends = defaultdict(lambda: {
            "accuracy_trend": deque(maxlen=24),  # Last 24 hours
            "latency_trend": deque(maxlen=24),
            "error_trend": deque(maxlen=24),
            "confidence_scores": deque(maxlen=100)
        })
        
        # Threat intelligence
        self.threat_patterns = {}
        self.suspicious_patterns = deque(maxlen=1000)
        
        # Performance baselines
        self.baselines = {
            "avg_latency": 0.0,
            "block_rate": 0.0,
            "error_rate": 0.0
        }
        
        # Anomaly detection
        self.anomalies = deque(maxlen=50)
        
        logger.info("Advanced metrics collector initialized")
    
    def record_advanced_analysis(self, prompt: str, user_id: Optional[str], 
                               final_verdict: str, guard_results: Dict, 
                               latency: float, request_time: float = None):
        """Record advanced metrics for analysis"""
        if request_time is None:
            request_time = time.time()
            
        # Update time-based analytics
        self._update_time_analytics(final_verdict, latency, request_time)
        
        # Update user behavior analytics
        if user_id:
            self._update_user_analytics(user_id, prompt, final_verdict, guard_results, request_time)
        
        # Update model performance trends
        self._update_model_trends(guard_results, latency, request_time)
        
        # Detect and record threats
        self._analyze_threat_patterns(prompt, final_verdict, guard_results)
        
        # Anomaly detection
        self._detect_anomalies(final_verdict, latency, request_time)
    
    def _update_time_analytics(self, verdict: str, latency: float, timestamp: float):
        """Update hourly and daily statistics"""
        hour_key = int(timestamp // 3600) * 3600
        day_key = int(timestamp // 86400) * 86400
        
        # Hourly stats
        hourly = self.hourly_stats[hour_key]
        hourly["requests"] += 1
        hourly[verdict + "s"] += 1
        hourly["total_latency"] += latency
        hourly["avg_latency"] = hourly["total_latency"] / hourly["requests"]
        
        # Daily stats
        daily = self.daily_stats[day_key]
        daily["requests"] += 1
        daily[verdict + "s"] += 1
        
        # Update peak hour
        current_hour = int((timestamp % 86400) // 3600)
        if hourly["requests"] > daily["peak_requests"]:
            daily["peak_hour"] = current_hour
            daily["peak_requests"] = hourly["requests"]
    
    def _update_user_analytics(self, user_id: str, prompt: str, verdict: str, 
                             guard_results: Dict, timestamp: float):
        """Update user behavior patterns"""
        user = self.user_patterns[user_id]
        user["total_requests"] += 1
        user["request_times"].append(timestamp)
        
        # Update rates
        total = user["total_requests"]
        if verdict == "block":
            user["block_rate"] = ((user["block_rate"] * (total - 1)) + 1) / total
        elif verdict == "warn":
            user["warn_rate"] = ((user["warn_rate"] * (total - 1)) + 1) / total
        
        # Update average prompt length
        current_avg = user["avg_prompt_length"]
        user["avg_prompt_length"] = ((current_avg * (total - 1)) + len(prompt)) / total
        
        # Update common categories
        for guard_result in guard_results.values():
            for label in guard_result.get("labels", []):
                user["common_categories"][label] += 1
        
        # Calculate risk score
        user["risk_score"] = self._calculate_user_risk_score(user)
    
    def _update_model_trends(self, guard_results: Dict, latency: float, timestamp: float):
        """Update model performance trends"""
        hour_bucket = int(timestamp // 3600) * 3600
        
        for guard_name, result in guard_results.items():
            trends = self.model_trends[guard_name]
            
            # Accuracy trend (simplified - based on confidence)
            confidence = result.get("score", 0.5)
            trends["confidence_scores"].append(confidence)
            
            # Latency trend (estimated per-guard)
            guard_latency = latency / len(guard_results)
            trends["latency_trend"].append({"timestamp": hour_bucket, "latency": guard_latency})
            
            # Error trend
            is_error = result.get("verdict") == "error"
            trends["error_trend"].append({"timestamp": hour_bucket, "error": is_error})
    
    def _analyze_threat_patterns(self, prompt: str, verdict: str, guard_results: Dict):
        """Analyze and record threat patterns"""
        if verdict in ["block", "warn"]:
            # Extract pattern indicators
            pattern_indicators = []
            affected_guards = []
            
            for guard_name, result in guard_results.items():
                if result.get("verdict") in ["block", "warn"]:
                    affected_guards.append(guard_name)
                    pattern_indicators.extend(result.get("labels", []))
            
            if pattern_indicators:
                pattern_key = "|".join(sorted(set(pattern_indicators)))
                
                if pattern_key in self.threat_patterns:
                    threat = self.threat_patterns[pattern_key]
                    threat.frequency += 1
                    threat.last_seen = time.time()
                    threat.affected_guards = list(set(threat.affected_guards + affected_guards))
                else:
                    self.threat_patterns[pattern_key] = ThreatIntelligence(
                        pattern=pattern_key,
                        severity="high" if verdict == "block" else "medium",
                        frequency=1,
                        first_seen=time.time(),
                        last_seen=time.time(),
                        affected_guards=affected_guards
                    )
    
    def _detect_anomalies(self, verdict: str, latency: float, timestamp: float):
        """Detect performance and security anomalies"""
        # Latency anomaly detection
        if self.baselines["avg_latency"] > 0:
            if latency > self.baselines["avg_latency"] * 3:  # 3x baseline
                self.anomalies.append({
                    "type": "latency_spike",
                    "timestamp": timestamp,
                    "value": latency,
                    "baseline": self.baselines["avg_latency"],
                    "severity": "high" if latency > self.baselines["avg_latency"] * 5 else "medium"
                })
        
        # Update baselines (exponential moving average)
        alpha = 0.1  # Smoothing factor
        if self.baselines["avg_latency"] == 0:
            self.baselines["avg_latency"] = latency
        else:
            self.baselines["avg_latency"] = alpha * latency + (1 - alpha) * self.baselines["avg_latency"]
    
    def _calculate_user_risk_score(self, user_data: Dict) -> float:
        """Calculate user risk score based on behavior patterns"""
        score = 0.0
        
        # High block rate increases risk
        score += user_data["block_rate"] * 40
        
        # High warn rate increases risk
        score += user_data["warn_rate"] * 20
        
        # Frequent requests in short time increases risk
        recent_requests = [t for t in user_data["request_times"] 
                          if time.time() - t < 3600]  # Last hour
        if len(recent_requests) > 50:  # More than 50 requests per hour
            score += 20
        
        # Normalize to 0-100 scale
        return min(score, 100.0)
    
    def get_time_based_analytics(self, window: TimeWindow = TimeWindow.DAY) -> Dict[str, Any]:
        """Get time-based analytics for specified window"""
        current_time = time.time()
        
        if window == TimeWindow.HOUR:
            return self._get_hourly_analytics(current_time)
        elif window == TimeWindow.DAY:
            return self._get_daily_analytics(current_time)
        elif window == TimeWindow.WEEK:
            return self._get_weekly_analytics(current_time)
        else:  # MONTH
            return self._get_monthly_analytics(current_time)
    
    def _get_hourly_analytics(self, current_time: float) -> Dict[str, Any]:
        """Get last 24 hours analytics"""
        cutoff = current_time - 86400  # 24 hours ago
        relevant_hours = {k: v for k, v in self.hourly_stats.items() if k >= cutoff}
        
        if not relevant_hours:
            return {"message": "No data available for the last 24 hours"}
        
        total_requests = sum(h["requests"] for h in relevant_hours.values())
        total_blocks = sum(h["blocks"] for h in relevant_hours.values())
        total_warns = sum(h["warns"] for h in relevant_hours.values())
        latency_values = [h["avg_latency"] for h in relevant_hours.values() if h["avg_latency"] > 0]
        avg_latency = statistics.mean(latency_values) if latency_values else 0
        
        return {
            "window": "24h",
            "total_requests": total_requests,
            "block_rate": (total_blocks / max(total_requests, 1)) * 100,
            "warn_rate": (total_warns / max(total_requests, 1)) * 100,
            "avg_latency_ms": avg_latency * 1000,
            "peak_hour": max(relevant_hours.items(), key=lambda x: x[1]["requests"])[0] if relevant_hours else None,
            "hourly_breakdown": [
                {
                    "hour": datetime.fromtimestamp(k).strftime("%H:00"),
                    "requests": v["requests"],
                    "blocks": v["blocks"],
                    "warns": v["warns"],
                    "avg_latency_ms": v["avg_latency"] * 1000
                }
                for k, v in sorted(relevant_hours.items())
            ]
        }
    
    def _get_daily_analytics(self, current_time: float) -> Dict[str, Any]:
        """Get last 7 days analytics"""
        cutoff = current_time - 604800  # 7 days ago
        relevant_days = {k: v for k, v in self.daily_stats.items() if k >= cutoff}
        
        if not relevant_days:
            return {"message": "No data available for the last 7 days"}
        
        return {
            "window": "7d",
            "daily_breakdown": [
                {
                    "date": datetime.fromtimestamp(k).strftime("%Y-%m-%d"),
                    "requests": v["requests"],
                    "blocks": v["blocks"],
                    "warns": v["warns"],
                    "unique_users": len(v["unique_users"]),
                    "peak_hour": f"{v['peak_hour']:02d}:00"
                }
                for k, v in sorted(relevant_days.items())
            ]
        }
    
    def _get_weekly_analytics(self, current_time: float) -> Dict[str, Any]:
        """Get last 4 weeks analytics"""
        # Simplified weekly aggregation
        return {"message": "Weekly analytics - aggregated from daily data"}
    
    def _get_monthly_analytics(self, current_time: float) -> Dict[str, Any]:
        """Get last 12 months analytics"""
        # Simplified monthly aggregation
        return {"message": "Monthly analytics - aggregated from daily data"}
    
    def get_user_behavior_analytics(self, limit: int = 20) -> Dict[str, Any]:
        """Get user behavior analytics"""
        # Sort users by risk score
        sorted_users = sorted(
            self.user_patterns.items(),
            key=lambda x: x[1]["risk_score"],
            reverse=True
        )[:limit]
        
        return {
            "high_risk_users": [
                {
                    "user_id": user_id,
                    "risk_score": data["risk_score"],
                    "total_requests": data["total_requests"],
                    "block_rate": data["block_rate"] * 100,
                    "warn_rate": data["warn_rate"] * 100,
                    "avg_prompt_length": data["avg_prompt_length"],
                    "top_categories": dict(sorted(
                        data["common_categories"].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5])
                }
                for user_id, data in sorted_users if data["risk_score"] > 30
            ],
            "user_statistics": {
                "total_users": len(self.user_patterns),
                "high_risk_users": len([u for u in self.user_patterns.values() if u["risk_score"] > 50]),
                "avg_requests_per_user": statistics.mean([u["total_requests"] for u in self.user_patterns.values()]) if self.user_patterns else 0.0
            }
        }
    
    def get_threat_intelligence(self) -> Dict[str, Any]:
        """Get threat intelligence and patterns"""
        # Sort threats by frequency and recency
        sorted_threats = sorted(
            self.threat_patterns.values(),
            key=lambda x: (x.frequency, x.last_seen),
            reverse=True
        )
        
        return {
            "active_threats": [
                {
                    "pattern": threat.pattern,
                    "severity": threat.severity,
                    "frequency": threat.frequency,
                    "first_seen": datetime.fromtimestamp(threat.first_seen).isoformat(),
                    "last_seen": datetime.fromtimestamp(threat.last_seen).isoformat(),
                    "affected_guards": threat.affected_guards
                }
                for threat in sorted_threats[:20]  # Top 20 threats
            ],
            "threat_summary": {
                "total_patterns": len(self.threat_patterns),
                "high_severity": len([t for t in self.threat_patterns.values() if t.severity == "high"]),
                "recent_threats": len([t for t in self.threat_patterns.values() 
                                     if time.time() - t.last_seen < 3600])  # Last hour
            }
        }
    
    def get_model_performance_trends(self) -> Dict[str, Any]:
        """Get model performance trends"""
        performance_data = {}
        
        for guard_name, trends in self.model_trends.items():
            # Calculate trend metrics
            recent_latencies = [item["latency"] for item in list(trends["latency_trend"])[-12:] if isinstance(item, dict)]  # Last 12 hours
            recent_errors = [item["error"] for item in list(trends["error_trend"])[-12:] if isinstance(item, dict)]
            recent_confidence = [score for score in list(trends["confidence_scores"])[-50:] if score is not None]  # Last 50 requests
            
            performance_data[guard_name] = {
                "avg_latency_trend": statistics.mean(recent_latencies) if recent_latencies else 0,
                "error_rate_trend": (sum(recent_errors) / len(recent_errors)) * 100 if recent_errors else 0,
                "avg_confidence": statistics.mean(recent_confidence) if recent_confidence else 0,
                "confidence_stability": statistics.stdev(recent_confidence) if len(recent_confidence) > 1 else 0,
                "trend_direction": self._calculate_trend_direction(recent_latencies)
            }
        
        return {
            "guard_performance": performance_data,
            "overall_health": self._calculate_overall_health(performance_data)
        }
    
    def _calculate_trend_direction(self, values: List[float]) -> str:
        """Calculate if trend is improving, degrading, or stable"""
        if len(values) < 2:
            return "stable"
        
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)
        
        if second_avg > first_avg * 1.1:
            return "degrading"
        elif second_avg < first_avg * 0.9:
            return "improving"
        else:
            return "stable"
    
    def _calculate_overall_health(self, performance_data: Dict) -> str:
        """Calculate overall system health"""
        if not performance_data:
            return "unknown"
        
        degrading_guards = sum(1 for p in performance_data.values() 
                             if p["trend_direction"] == "degrading" or p["error_rate_trend"] > 5)
        
        total_guards = len(performance_data)
        degrading_ratio = degrading_guards / total_guards
        
        if degrading_ratio > 0.5:
            return "critical"
        elif degrading_ratio > 0.25:
            return "degraded"
        else:
            return "healthy"
    
    def get_anomalies(self) -> List[Dict[str, Any]]:
        """Get recent anomalies"""
        return [
            {
                "type": anomaly["type"],
                "timestamp": datetime.fromtimestamp(anomaly["timestamp"]).isoformat(),
                "severity": anomaly["severity"],
                "details": {
                    k: v for k, v in anomaly.items() 
                    if k not in ["type", "timestamp", "severity"]
                }
            }
            for anomaly in list(self.anomalies)
        ]
    
    def get_comprehensive_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            model_performance = self.get_model_performance_trends()
            return {
                "time_analytics": self.get_time_based_analytics(TimeWindow.DAY),
                "user_behavior": self.get_user_behavior_analytics(),
                "threat_intelligence": self.get_threat_intelligence(),
                "model_performance": model_performance,
                "recent_anomalies": self.get_anomalies()[-10:],  # Last 10 anomalies
                "system_health": {
                    "overall_status": self._calculate_overall_health(
                        model_performance.get("guard_performance", {})
                    ),
                    "active_threats": len(self.threat_patterns),
                    "high_risk_users": len([u for u in self.user_patterns.values() if u["risk_score"] > 50]),
                    "recent_anomalies": len([a for a in self.anomalies if time.time() - a["timestamp"] < 3600])
                }
            }
        except Exception as e:
            logger.error(f"Error generating comprehensive dashboard: {e}")
            return {
                "error": "Failed to generate dashboard data",
                "message": str(e),
                "basic_stats": {
                    "total_users": len(self.user_patterns),
                    "active_threats": len(self.threat_patterns),
                    "anomalies": len(self.anomalies)
                }
            }