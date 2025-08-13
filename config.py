from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class Config:
    """Configuration management for guards and thresholds"""
    
    def __init__(self):
        # Master toggle
        self.enable_all = True
        
        # Individual guard toggles
        self.guards = {
            "llama_guard_8b": True,
            "llama_guard_1b": True,
            "indobert_toxic": True,
            "llm_guard": True,
            "nemo_guardrails": True
        }
        
        # Adjustable thresholds
        self.thresholds = {
            "indobert_threshold": 0.70,  # IndoBERT toxicity warning threshold
            "llm_guard_sensitivity": 0.8,  # LLM Guard sensitivity
            "nemo_confidence": 0.75,  # NeMo Guardrails confidence threshold
        }
        
        logger.info("Configuration initialized with default values")
    
    def get_enabled_guards(self) -> List[str]:
        """Get list of currently enabled guards"""
        if not self.enable_all:
            return []
            
        return [name for name, enabled in self.guards.items() if enabled]
    
    def get_thresholds(self) -> Dict[str, float]:
        """Get current threshold values"""
        return self.thresholds.copy()
    
    def update_guard_status(self, guard_name: str, enabled: bool) -> bool:
        """Update individual guard status"""
        if guard_name in self.guards:
            self.guards[guard_name] = enabled
            logger.info(f"Guard {guard_name} {'enabled' if enabled else 'disabled'}")
            return True
        return False
    
    def update_threshold(self, threshold_name: str, value: float) -> bool:
        """Update threshold value"""
        if threshold_name in self.thresholds:
            # Validate threshold range
            if 0.0 <= value <= 1.0:
                self.thresholds[threshold_name] = value
                logger.info(f"Threshold {threshold_name} updated to {value}")
                return True
            else:
                logger.warning(f"Invalid threshold value {value} for {threshold_name}")
        return False
    
    def toggle_all_guards(self, enabled: bool):
        """Enable or disable all guards"""
        self.enable_all = enabled
        logger.info(f"All guards {'enabled' if enabled else 'disabled'}")
    
    def get_config_summary(self) -> Dict:
        """Get complete configuration summary"""
        return {
            "enable_all": self.enable_all,
            "enabled_guards": self.get_enabled_guards(),
            "guard_status": self.guards.copy(),
            "thresholds": self.thresholds.copy(),
            "total_guards": len(self.guards),
            "active_guards": len(self.get_enabled_guards())
        }
    
    def reset_to_defaults(self):
        """Reset configuration to default values"""
        self.__init__()
        logger.info("Configuration reset to defaults")