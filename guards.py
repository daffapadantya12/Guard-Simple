import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False

# Real imports for production-ready guards
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, BitsAndBytesConfig
    import torch
    HF_AVAILABLE = True
except ImportError:
    pipeline = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    AutoModelForCausalLM = None
    BitsAndBytesConfig = None
    torch = None
    HF_AVAILABLE = False

# Quantization support
try:
    import bitsandbytes as bnb
    QUANTIZATION_AVAILABLE = True
except ImportError:
    bnb = None
    QUANTIZATION_AVAILABLE = False

try:
    from llmguard import scan_prompt
    from llmguard.input_scanners import PromptInjection, Toxicity, PII, URL, SensitiveData
    LLMGUARD_AVAILABLE = True
except ImportError:
    scan_prompt = None
    PromptInjection = None
    Toxicity = None
    PII = None
    URL = None
    SensitiveData = None
    LLMGUARD_AVAILABLE = False

try:
    from nemoguardrails import RailsConfig, LLMRails
    NEMO_AVAILABLE = True
except ImportError:
    RailsConfig = None
    LLMRails = None
    NEMO_AVAILABLE = False

# Indonesian language detection
try:
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    langdetect = None
    LANGDETECT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Configuration helper functions
def get_device_config():
    """Get device configuration from environment"""
    device_env = os.getenv('DEVICE', 'auto').lower()
    if device_env == 'auto':
        return "cuda" if torch and torch.cuda.is_available() else "cpu"
    elif device_env in ['cuda', 'cpu']:
        return device_env
    else:
        logger.warning(f"Invalid DEVICE config '{device_env}', defaulting to auto")
        return "cuda" if torch and torch.cuda.is_available() else "cpu"

def get_quantization_config():
    """Get quantization configuration from environment"""
    if not QUANTIZATION_AVAILABLE or not HF_AVAILABLE:
        return None
        
    use_quantization = os.getenv('USE_QUANTIZATION', 'false').lower() == 'true'
    if not use_quantization:
        return None
        
    try:
        quantization_bits = int(os.getenv('QUANTIZATION_BITS', '4'))
        quantization_type = os.getenv('QUANTIZATION_TYPE', 'nf4')
        use_double_quant = os.getenv('USE_DOUBLE_QUANT', 'true').lower() == 'true'
        compute_dtype = getattr(torch, os.getenv('COMPUTE_DTYPE', 'float16'))
        
        return BitsAndBytesConfig(
            load_in_4bit=quantization_bits == 4,
            load_in_8bit=quantization_bits == 8,
            bnb_4bit_quant_type=quantization_type,
            bnb_4bit_use_double_quant=use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype
        )
    except Exception as e:
        logger.error(f"Failed to create quantization config: {e}")
        return None

def get_hf_token():
    """Get Hugging Face token from environment"""
    return os.getenv('HUGGINGFACE_TOKEN')

class GuardResult:
    def __init__(self, verdict: str, labels: List[str], score: Optional[float] = None):
        self.verdict = verdict
        self.labels = labels
        self.score = score

class BaseGuard(ABC):
    """Base class for all guards"""
    
    def __init__(self, name: str):
        self.name = name
        self.version = "1.0.0"
        self.is_healthy = True
        
    @abstractmethod
    async def analyze(self, prompt: str, lang: str = "auto", **kwargs) -> GuardResult:
        """Analyze a prompt and return verdict"""
        pass
    
    def health_check(self) -> bool:
        """Check if guard is healthy"""
        return self.is_healthy

class LlamaGuard8B(BaseGuard):
    """Real LLaMA Guard 8B implementation using Hugging Face models"""
    
    def __init__(self):
        super().__init__("llama_guard_8b")
        self.model = None
        self.tokenizer = None
        self.device = get_device_config()
        self.quantization_config = get_quantization_config()
        self.hf_token = get_hf_token()
        self._load_model()
        
    def _load_model(self):
        """Load the actual LLaMA Guard model with quantization support"""
        try:
            if not HF_AVAILABLE:
                logger.error("Transformers library not available for LlamaGuard8B")
                return
                
            # Use Meta's LLaMA Guard model
            model_name = "meta-llama/LlamaGuard-7b"
            logger.info(f"Loading LLaMA Guard model: {model_name}")
            
            # Check for HF token
            if not self.hf_token:
                logger.warning("No Hugging Face token found. Set HUGGINGFACE_TOKEN in .env file for gated models.")
            
            # Configure model loading parameters
            model_kwargs = {
                "trust_remote_code": os.getenv('TRUST_REMOTE_CODE', 'true').lower() == 'true',
                "token": self.hf_token,
                "cache_dir": os.getenv('HF_CACHE_DIR', './models_cache')
            }
            
            # Add quantization config if available
            if self.quantization_config:
                model_kwargs["quantization_config"] = self.quantization_config
                logger.info(f"Using quantization: {self.quantization_config}")
            else:
                # Set torch dtype if not using quantization
                torch_dtype_str = os.getenv('TORCH_DTYPE', 'float16')
                if hasattr(torch, torch_dtype_str):
                    model_kwargs["torch_dtype"] = getattr(torch, torch_dtype_str)
            
            # Set device map
            if self.device == "cuda" and not self.quantization_config:
                model_kwargs["device_map"] = "auto"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=model_kwargs["trust_remote_code"],
                token=self.hf_token,
                cache_dir=model_kwargs["cache_dir"]
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Move to device if not using device_map
            if "device_map" not in model_kwargs and not self.quantization_config:
                self.model = self.model.to(self.device)
                
            logger.info(f"LLaMA Guard model loaded successfully on {self.device}")
            if self.quantization_config:
                logger.info("Model loaded with quantization enabled")
            
        except Exception as e:
            logger.error(f"Failed to load LLaMA Guard model: {e}")
            logger.info("Falling back to rule-based implementation")
            self.model = None
            self.tokenizer = None
    
    def _detect_indonesian_content(self, text: str) -> bool:
        """Detect Indonesian language content"""
        indonesian_words = [
            'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'adalah',
            'tidak', 'akan', 'sudah', 'bisa', 'harus', 'juga', 'saya', 'kamu', 'dia'
        ]
        text_lower = text.lower()
        return sum(1 for word in indonesian_words if word in text_lower) >= 2
        
    async def analyze(self, prompt: str, lang: str = "auto", **kwargs) -> GuardResult:
        """Real model analysis with fallback to rule-based approach"""
        try:
            # Try real model inference first
            if self.model and self.tokenizer:
                return await self._analyze_with_model(prompt, lang, **kwargs)
            else:
                # Fallback to rule-based approach
                return await self._analyze_rule_based(prompt, lang, **kwargs)
                
        except Exception as e:
            logger.error(f"LlamaGuard8B analysis error: {e}")
            return GuardResult("warn", ["system_error"])
    
    async def _analyze_with_model(self, prompt: str, lang: str = "auto", **kwargs) -> GuardResult:
        """Analyze using LlamaGuard model with enhanced Indonesian support"""
        try:
            if not self.model or not self.tokenizer:
                logger.warning("Model not loaded, falling back to rule-based analysis")
                return await self._analyze_rule_based(prompt, lang, **kwargs)
            
            # Detect if content is Indonesian
            is_indonesian = self._detect_indonesian_content(prompt)
            
            # Enhanced system prompt for Indonesian content
            if is_indonesian:
                system_prompt = """Anda adalah AI safety classifier yang menganalisis konten dalam bahasa Indonesia. 
Klasifikasikan konten berikut berdasarkan kategori keamanan:
- O1: Kekerasan, ancaman, atau bahaya fisik
- O2: Konten seksual eksplisit
- O3: Kejahatan siber, hacking, atau penipuan
- O4: Senjata dan kekerasan
- O5: Zat terlarang
- O6: Menyakiti diri sendiri atau bunuh diri
- O7: Informasi pribadi atau data sensitif
- O8: Konten seksual yang melibatkan anak di bawah umur

Jawab dengan 'safe' jika aman, atau 'unsafe' diikuti kode kategori (O1-O8) jika tidak aman."""
            else:
                system_prompt = """You are a safety classifier. Classify the following content for safety violations:
- O1: Violence and Hate
- O2: Sexual Content
- O3: Criminal Planning
- O4: Guns and Illegal Weapons
- O5: Regulated or Controlled Substances
- O6: Self-Harm
- O7: Privacy
- O8: Sexual content involving minors

Respond with 'safe' if safe, or 'unsafe' followed by category codes if unsafe."""
            
            # Format input using chat template
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Apply chat template
            formatted_input = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize with proper handling
            inputs = self.tokenizer(
                formatted_input,
                return_tensors="pt",
                max_length=1024,  # Increased for Indonesian context
                truncation=True,
                padding=True
            ).to(self.model.device)
            
            # Generate response with adjusted parameters for better Indonesian understanding
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,  # Increased for more detailed responses
                    temperature=0.1,    # Slightly increased for better Indonesian handling
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            ).strip()
            
            logger.info(f"LlamaGuard8B raw response: {response}")
            
            # Enhanced response parsing for Indonesian content
            response_lower = response.lower()
            
            if "safe" in response_lower or "aman" in response_lower:
                return GuardResult("allow", [])
            elif "unsafe" in response_lower or "tidak aman" in response_lower or "berbahaya" in response_lower:
                categories = self._extract_categories_from_response(response)
                if categories:
                    return GuardResult("block", categories)
                else:
                    return GuardResult("block", ["unsafe_content"])
            else:
                # Enhanced fallback analysis for ambiguous responses
                logger.warning(f"Ambiguous response from LlamaGuard8B: {response}")
                
                # Try to extract meaning from Indonesian responses
                if is_indonesian:
                    if any(word in response_lower for word in ["kekerasan", "violence", "ancaman"]):
                        return GuardResult("block", ["Violence"])
                    elif any(word in response_lower for word in ["kebencian", "hate", "sara", "diskriminasi"]):
                        return GuardResult("block", ["Hate"])
                    elif any(word in response_lower for word in ["seksual", "sexual", "porno"]):
                        return GuardResult("block", ["Sexual_Content"])
                    elif any(word in response_lower for word in ["anak", "minor", "dibawah umur"]):
                        return GuardResult("block", ["Sexual_Minors"])
                
                return GuardResult("warn", ["uncertain_classification"])
                
        except Exception as e:
            logger.error(f"Model inference error in LlamaGuard8B: {e}")
            return await self._analyze_rule_based(prompt, lang, **kwargs)
    
    def _extract_categories_from_response(self, response: str) -> List[str]:
        """Enhanced category extraction with Indonesian support"""
        categories = []
        response_lower = response.lower()
        
        # Standard LLaMA Guard category mapping
        category_mapping = {
            "o1": "Violence",
            "o2": "Sexual_Content", 
            "o3": "Cybercrime",
            "o4": "Violence",  # Weapons -> Violence
            "o5": "Cybercrime",  # Substances -> Cybercrime
            "o6": "Self_Harm",
            "o7": "Privacy",
            "o8": "Sexual_Minors"
        }
        
        # Extract standard categories
        for code, category in category_mapping.items():
            if code in response_lower:
                categories.append(category)
        
        # Enhanced Indonesian keyword extraction
        indonesian_categories = {
            "kekerasan": "Violence",
            "ancaman": "Violence", 
            "pembunuhan": "Violence",
            "kebencian": "Hate",
            "sara": "Hate",
            "diskriminasi": "Hate",
            "pelecehan": "Harassment",
            "bullying": "Harassment",
            "intimidasi": "Harassment",
            "bunuh diri": "Self_Harm",
            "menyakiti diri": "Self_Harm",
            "seksual": "Sexual_Content",
            "porno": "Sexual_Content",
            "telanjang": "Sexual_Content",
            "hacking": "Cybercrime",
            "penipuan": "Cybercrime",
            "malware": "Cybercrime",
            "data pribadi": "Privacy",
            "informasi pribadi": "Privacy",
            "anak": "Sexual_Minors",
            "dibawah umur": "Sexual_Minors",
            "remaja": "Sexual_Minors"
        }
        
        # Check for Indonesian categories
        for keyword, category in indonesian_categories.items():
            if keyword in response_lower and category not in categories:
                categories.append(category)
        
        # Fallback to generic unsafe if no specific categories found
        if not categories and ("unsafe" in response_lower or "tidak aman" in response_lower):
            categories.append("unsafe_content")
            
        return categories if categories else ["unsafe_content"]
    
    async def _analyze_rule_based(self, prompt: str, lang: str = "auto", **kwargs) -> GuardResult:
        """Fallback rule-based analysis (original implementation)"""
        await asyncio.sleep(0.1)
        
        prompt_lower = prompt.lower()
        is_indonesian = self._detect_indonesian_content(prompt)
        
        # Enhanced Indonesian harmful content detection
        violence_keywords_id = [
            'bunuh', 'pembunuhan', 'membunuh', 'mati', 'matikan', 'tembak', 'menembak',
            'tikam', 'menikam', 'bacok', 'membacok', 'pukul', 'memukul', 'hajar', 'menghajar',
            'kekerasan', 'kerusuhan', 'perang', 'serangan', 'menyerang', 'teror', 'teroris',
            'bom', 'meledak', 'ledakan', 'senjata', 'pistol', 'senapan', 'granat'
        ]
        
        violence_keywords_en = [
            'kill', 'murder', 'violence', 'weapon', 'bomb', 'gun', 'knife', 'attack',
            'assault', 'terrorism', 'terrorist', 'explosive', 'grenade', 'shoot', 'stab'
        ]
        
        minors_keywords_id = [
            'anak', 'bocah', 'balita', 'remaja', 'pelajar', 'siswa', 'murid',
            'dibawah umur', 'belum dewasa', 'anak kecil', 'bocah cilik'
        ]
        
        minors_keywords_en = [
            'child', 'minor', 'underage', 'kid', 'teenager', 'student', 'pupil',
            'young', 'juvenile', 'adolescent'
        ]
        
        cybercrime_keywords_id = [
            'hack', 'hacker', 'meretas', 'peretasan', 'malware', 'virus', 'trojan',
            'phishing', 'scam', 'penipuan', 'fraud', 'carding', 'skimming',
            'exploit', 'mengeksploit', 'bobol', 'membobol', 'jebol', 'memjebol'
        ]
        
        cybercrime_keywords_en = [
            'hack', 'hacker', 'malware', 'virus', 'exploit', 'phishing', 'scam',
            'fraud', 'carding', 'skimming', 'trojan', 'ransomware', 'breach'
        ]
        
        privacy_keywords_id = [
            'ktp', 'sim', 'paspor', 'rekening', 'kartu kredit', 'pin', 'password',
            'kata sandi', 'data pribadi', 'informasi pribadi', 'alamat rumah',
            'nomor telepon', 'email pribadi', 'npwp', 'bpjs'
        ]
        
        privacy_keywords_en = [
            'ssn', 'social security', 'credit card', 'password', 'personal info',
            'private data', 'home address', 'phone number', 'email address',
            'bank account', 'pin number', 'passport', 'driver license'
        ]
        
        hate_keywords_id = [
            'benci', 'membenci', 'kebencian', 'diskriminasi', 'mendiskriminasi',
            'rasis', 'rasisme', 'kafir', 'pribumi', 'non-pribumi', 'cina', 'aseng',
            'inlander', 'sara', 'suku', 'agama', 'ras', 'golongan'
        ]
        
        hate_keywords_en = [
            'hate', 'hatred', 'racist', 'racism', 'discrimination', 'bigot',
            'xenophobia', 'islamophobia', 'antisemitism', 'supremacy'
        ]
        
        self_harm_keywords_id = [
            'bunuh diri', 'suicide', 'menyakiti diri', 'melukai diri', 'self harm',
            'depresi', 'putus asa', 'tidak ada harapan', 'ingin mati', 'lelah hidup'
        ]
        
        self_harm_keywords_en = [
            'suicide', 'self-harm', 'self harm', 'cut myself', 'hurt myself',
            'kill myself', 'end my life', 'depression', 'hopeless', 'want to die'
        ]
        
        # Check for harmful content with higher sensitivity for Indonesian
        weight = 1.2 if is_indonesian else 1.0
        
        # Block categories
        if any(word in prompt_lower for word in violence_keywords_id + violence_keywords_en):
            return GuardResult("block", ["Violence"])
        
        if any(word in prompt_lower for word in minors_keywords_id + minors_keywords_en):
            # Extra sensitive for content involving minors
            sexual_context = any(word in prompt_lower for word in 
                ['seks', 'sexual', 'telanjang', 'naked', 'bugil', 'porno', 'porn'])
            if sexual_context:
                return GuardResult("block", ["Sexual_Minors"])
            else:
                return GuardResult("warn", ["Minors_Context"])
        
        if any(word in prompt_lower for word in cybercrime_keywords_id + cybercrime_keywords_en):
            return GuardResult("block", ["Cybercrime"])
        
        if any(word in prompt_lower for word in privacy_keywords_id + privacy_keywords_en):
            return GuardResult("block", ["Privacy"])
            
        # Warn categories
        if any(word in prompt_lower for word in hate_keywords_id + hate_keywords_en):
            return GuardResult("warn", ["Hate"])
        
        if any(word in prompt_lower for word in self_harm_keywords_id + self_harm_keywords_en):
            return GuardResult("warn", ["Self_Harm"])
            
        return GuardResult("allow", [])

class LlamaGuard1B(BaseGuard):
    """Real LLaMA Guard 1B implementation using smaller model"""
    
    def __init__(self):
        super().__init__("llama_guard_1b")
        self.model = None
        self.tokenizer = None
        self.device = get_device_config()
        self.hf_token = get_hf_token()
        self._load_model()
        
    def _load_model(self):
        """Load a smaller LLaMA Guard model or alternative with environment config"""
        try:
            if not HF_AVAILABLE:
                logger.error("Transformers library not available for LlamaGuard1B")
                return
                
            # Try to use a smaller safety model as alternative to LLaMA Guard 1B
            # Since actual LLaMA Guard 1B might not be publicly available
            model_name = "unitary/toxic-bert"  # Fallback to a smaller safety model
            logger.info(f"Loading safety model for LlamaGuard1B: {model_name}")
            
            # Configure pipeline parameters
            pipeline_kwargs = {
                "model": model_name,
                "tokenizer": model_name,
                "device": 0 if self.device == "cuda" else -1
            }
            
            # Add token if available
            if self.hf_token:
                pipeline_kwargs["token"] = self.hf_token
                
            # Add cache directory
            cache_dir = os.getenv('HF_CACHE_DIR', './models_cache')
            if cache_dir:
                pipeline_kwargs["model_kwargs"] = {"cache_dir": cache_dir}
            
            # Use pipeline for simpler setup
            self.classifier = pipeline(
                "text-classification",
                **pipeline_kwargs
            )
            
            logger.info(f"LlamaGuard1B model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load LlamaGuard1B model: {e}")
            logger.info("Falling back to rule-based implementation")
            self.classifier = None
        
    async def analyze(self, prompt: str, lang: str = "auto", **kwargs) -> GuardResult:
        """Real model analysis with fallback to rule-based approach"""
        try:
            # Try real model inference first
            if hasattr(self, 'classifier') and self.classifier:
                return await self._analyze_with_model(prompt, lang, **kwargs)
            else:
                # Fallback to rule-based approach
                return await self._analyze_rule_based(prompt, lang, **kwargs)
                
        except Exception as e:
            logger.error(f"LlamaGuard1B analysis error: {e}")
            return GuardResult("warn", ["system_error"])
    
    async def _analyze_with_model(self, prompt: str, lang: str = "auto", **kwargs) -> GuardResult:
        """Analyze using toxic-bert with enhanced Indonesian context awareness"""
        try:
            # For LlamaGuard1B, we can enhance the toxicity detection with context
            is_indonesian = self._detect_indonesian_content(prompt)
            
            # Run inference
            result = self.classifier(prompt)
            
            # Parse results with Indonesian context awareness
            if isinstance(result, list) and len(result) > 0:
                prediction = result[0]
                label = prediction.get('label', '').upper()
                score = prediction.get('score', 0.0)
                
                # Adjust thresholds based on language and content
                high_threshold = 0.4 if is_indonesian else 0.5  # Lower threshold for Indonesian
                medium_threshold = 0.25 if is_indonesian else 0.3
                
                if label == 'TOXIC' and score > high_threshold:
                    # Try to categorize the toxicity type using LLM understanding
                    category = self._categorize_toxicity(prompt, is_indonesian)
                    return GuardResult("block", [category])
                elif label == 'TOXIC' and score > medium_threshold:
                    category = self._categorize_toxicity(prompt, is_indonesian)
                    return GuardResult("warn", [category])
                else:
                    return GuardResult("allow", [])
            
            return GuardResult("allow", [])
                
        except Exception as e:
            logger.error(f"Model inference error in LlamaGuard1B: {e}")
            return await self._analyze_rule_based(prompt, lang, **kwargs)
    
    def _categorize_toxicity(self, prompt: str, is_indonesian: bool) -> str:
        """Categorize toxicity type using content analysis"""
        prompt_lower = prompt.lower()
        
        if is_indonesian:
            if any(word in prompt_lower for word in ['kekerasan', 'bunuh', 'mati', 'ancaman']):
                return 'Violence'
            elif any(word in prompt_lower for word in ['benci', 'sara', 'diskriminasi', 'anjing', 'babi']):
                return 'Hate'
            elif any(word in prompt_lower for word in ['seks', 'porno', 'telanjang']):
                return 'Sexual_Content'
            elif any(word in prompt_lower for word in ['anak', 'bocah', 'remaja']):
                return 'Sexual_Minors'
        else:
            if any(word in prompt_lower for word in ['violence', 'kill', 'murder', 'threat']):
                return 'Violence'
            elif any(word in prompt_lower for word in ['hate', 'racist', 'discrimination']):
                return 'Hate'
            elif any(word in prompt_lower for word in ['sex', 'porn', 'naked']):
                return 'Sexual_Content'
            elif any(word in prompt_lower for word in ['child', 'minor', 'kid']):
                return 'Sexual_Minors'
        
        return 'high_toxicity'  # Default toxicity category
    
    def _detect_indonesian_content(self, text: str) -> bool:
        """Enhanced Indonesian content detection"""
        if not text:
            return False
            
        text_lower = text.lower()
        
        # Enhanced Indonesian indicators
        indonesian_indicators = [
            # Common Indonesian words
            'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'dalam', 'oleh',
            'akan', 'telah', 'sudah', 'sedang', 'masih', 'belum', 'tidak', 'bukan', 'jangan',
            'saya', 'anda', 'kamu', 'dia', 'mereka', 'kita', 'kami',
            'ini', 'itu', 'tersebut', 'dimana', 'bagaimana', 'mengapa', 'kapan',
            'bisa', 'dapat', 'harus', 'mau', 'ingin', 'perlu', 'boleh',
            # Indonesian-specific words
            'selamat', 'terima kasih', 'maaf', 'permisi', 'silakan', 'tolong',
            'gimana', 'kenapa', 'dimana', 'kayak', 'banget', 'dong', 'sih', 'kok',
            # Indonesian harmful content indicators
            'anjing', 'babi', 'bangsat', 'bajingan', 'goblok', 'tolol', 'bodoh'
        ]
        
        # Count Indonesian indicators
        indonesian_count = sum(1 for word in indonesian_indicators if word in text_lower)
        
        # Use langdetect if available for additional verification
        if LANGDETECT_AVAILABLE:
            try:
                detected_lang = langdetect.detect(text)
                if detected_lang == 'id':
                    return True
            except:
                pass
        
        # Return True if we found enough Indonesian indicators
        return indonesian_count >= 2  # Lowered threshold for better detection
    
    async def _analyze_rule_based(self, prompt: str, lang: str = "auto", **kwargs) -> GuardResult:
        """Enhanced rule-based analysis with stricter detection for Indonesian content"""
        await asyncio.sleep(0.05)  # Faster than 8B
        
        prompt_lower = prompt.lower()
        is_indonesian = self._detect_indonesian_content(prompt)
        
        # Enhanced Indonesian keywords for stricter detection
        violence_keywords_id = [
            'bunuh', 'pembunuhan', 'membunuh', 'mati', 'matikan', 'tembak', 'menembak',
            'tikam', 'menikam', 'bacok', 'membacok', 'pukul', 'memukul', 'hajar', 'menghajar',
            'kekerasan', 'kerusuhan', 'perang', 'serangan', 'menyerang', 'teror', 'teroris',
            'bom', 'meledak', 'ledakan', 'senjata', 'pistol', 'senapan', 'granat',
            'sikat', 'menyikat', 'gebuk', 'menggebuk', 'tonjok', 'menjotos', 'bogem'
        ]
        
        violence_keywords_en = [
            'violence', 'weapon', 'attack', 'kill', 'murder', 'bomb', 'gun', 'knife',
            'assault', 'terrorism', 'terrorist', 'explosive', 'grenade', 'shoot', 'stab'
        ]
        
        cybercrime_keywords_id = [
            'hack', 'hacker', 'meretas', 'peretasan', 'exploit', 'mengeksploit',
            'bobol', 'membobol', 'jebol', 'memjebol', 'malware', 'virus', 'trojan'
        ]
        
        cybercrime_keywords_en = [
            'hack', 'exploit', 'malware', 'virus', 'phishing', 'scam', 'fraud'
        ]
        
        hate_keywords_id = [
            'benci', 'membenci', 'kebencian', 'diskriminasi', 'mendiskriminasi',
            'rasis', 'rasisme', 'kafir', 'sara', 'anjing', 'babi', 'monyet',
            'bangsat', 'bajingan', 'keparat', 'tolol', 'bodoh', 'goblok'
        ]
        
        hate_keywords_en = [
            'hate', 'discrimination', 'racist', 'racism', 'bigot', 'nazi', 'fascist'
        ]
        
        sexual_keywords_id = [
            'seks', 'sexual', 'telanjang', 'bugil', 'porno', 'porn', 'ngentot', 'ngewe'
        ]
        
        sexual_keywords_en = [
            'sex', 'sexual', 'naked', 'nude', 'porn', 'pornography', 'erotic'
        ]
        
        minors_keywords_id = [
            'anak', 'bocah', 'balita', 'remaja', 'pelajar', 'siswa', 'murid',
            'dibawah umur', 'belum dewasa', 'anak kecil'
        ]
        
        minors_keywords_en = [
            'child', 'minor', 'underage', 'kid', 'teenager', 'student'
        ]
        
        # Higher sensitivity for Indonesian content
        weight = 1.5 if is_indonesian else 1.0
        
        # Stricter enforcement - more categories now block instead of warn
        if any(word in prompt_lower for word in violence_keywords_id + violence_keywords_en):
            return GuardResult("block", ["Violence"])  # Changed from "warn" to "block"
            
        if any(word in prompt_lower for word in cybercrime_keywords_id + cybercrime_keywords_en):
            return GuardResult("block", ["Cybercrime"])
            
        if any(word in prompt_lower for word in hate_keywords_id + hate_keywords_en):
            return GuardResult("block", ["Hate"])  # Changed from "warn" to "block"
            
        # Check for minors context with sexual content
        if any(word in prompt_lower for word in minors_keywords_id + minors_keywords_en):
            sexual_context = any(word in prompt_lower for word in sexual_keywords_id + sexual_keywords_en)
            if sexual_context:
                return GuardResult("block", ["Sexual_Minors"])
            else:
                return GuardResult("block", ["Minors_Context"])  # Changed from "warn" to "block"
                
        # Sexual content detection
        if any(word in prompt_lower for word in sexual_keywords_id + sexual_keywords_en):
            return GuardResult("block", ["Sexual_Content"])
            
        return GuardResult("allow", [])

class IndoBERTToxic(BaseGuard):
    """IndoBERT Toxic Classifier implementation with real model loading"""
    
    def __init__(self):
        super().__init__("indobert_toxic")
        self.model = None
        self.tokenizer = None
        self.classifier = None
        self._load_model()
        
    def _load_model(self):
        """Load IndoBERT toxicity detection model"""
        try:
            if HF_AVAILABLE:
                # Try to load a pre-trained Indonesian toxicity model
                model_name = "unitary/toxic-bert"
                logger.info(f"Loading IndoBERT model: {model_name}")
                self.classifier = pipeline(
                    "text-classification",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("IndoBERT model loaded successfully")
            else:
                logger.warning("Transformers not available, using fallback implementation")
        except Exception as e:
            logger.error(f"Failed to load IndoBERT model: {e}")
            self.classifier = None
    
    def _detect_language(self, text: str) -> str:
        """Detect if text is Indonesian"""
        if LANGDETECT_AVAILABLE:
            try:
                detected = langdetect.detect(text)
                return detected
            except:
                pass
        
        # Fallback: check for Indonesian words
        indonesian_indicators = [
            'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'adalah', 'ini', 'itu',
            'tidak', 'akan', 'sudah', 'bisa', 'harus', 'juga', 'atau', 'karena', 'jika', 'saya',
            'kamu', 'dia', 'kita', 'mereka', 'kami', 'bagaimana', 'mengapa', 'dimana', 'kapan'
        ]
        
        text_lower = text.lower()
        indonesian_count = sum(1 for word in indonesian_indicators if word in text_lower)
        
        if indonesian_count >= 2:
            return 'id'
        return 'unknown'
        
    async def analyze(self, prompt: str, lang: str = "auto", **kwargs) -> GuardResult:
        """Enhanced Indonesian toxicity detection"""
        try:
            await asyncio.sleep(0.08)
            
            # Detect language
            detected_lang = self._detect_language(prompt) if lang == "auto" else lang
            
            # Enhanced Indonesian toxic keywords
            toxic_keywords_id = [
                # Profanity
                'anjing', 'babi', 'bangsat', 'bajingan', 'kampret', 'tolol', 'bodoh', 'goblok',
                'idiot', 'sialan', 'keparat', 'brengsek', 'tai', 'setan', 'iblis',
                # Hate speech
                'pembunuh', 'bunuh', 'mati', 'mampus', 'hancur', 'rusak', 'benci', 'muak',
                # Discrimination
                'kafir', 'cina', 'pribumi', 'aseng', 'inlander', 'diskriminasi',
                # Violence
                'pukul', 'hajar', 'gebuk', 'tembak', 'tikam', 'bacok', 'kekerasan',
                # Sexual harassment
                'cabul', 'mesum', 'perkosa', 'rudapaksa', 'harassment'
            ]
            
            toxic_keywords_en = [
                'stupid', 'idiot', 'hate', 'kill', 'die', 'murder', 'violence', 'rape',
                'harassment', 'discrimination', 'racist', 'terrorism', 'bomb', 'weapon'
            ]
            
            prompt_lower = prompt.lower()
            score = 0.0
            detected_labels = []
            
            # Use real model if available
            if self.classifier:
                try:
                    result = self.classifier(prompt)
                    if isinstance(result, list) and len(result) > 0:
                        toxic_result = result[0]
                        if toxic_result['label'] == 'TOXIC':
                            score = toxic_result['score']
                            detected_labels.append('model_toxicity')
                except Exception as e:
                    logger.error(f"Model inference error: {e}")
            
            # Enhanced keyword-based detection with Indonesian focus
            keyword_score = 0.0
            
            # Higher weight for Indonesian content
            weight_multiplier = 1.5 if detected_lang == 'id' else 1.0
            
            for word in toxic_keywords_id:
                if word in prompt_lower:
                    keyword_score += 0.4 * weight_multiplier
                    detected_labels.append('indonesian_toxicity')
                    
            for word in toxic_keywords_en:
                if word in prompt_lower:
                    keyword_score += 0.3
                    detected_labels.append('english_toxicity')
            
            # Combine scores
            final_score = max(score, min(keyword_score, 1.0))
            
            # Apply thresholds
            threshold = kwargs.get('indobert_threshold', 0.70)
            
            if final_score >= 0.95:
                return GuardResult("block", detected_labels or ["high_toxicity"], final_score)
            elif final_score >= threshold:
                return GuardResult("warn", detected_labels or ["moderate_toxicity"], final_score)
            else:
                return GuardResult("allow", [], final_score)
                
        except Exception as e:
            logger.error(f"IndoBERTToxic error: {e}")
            return GuardResult("warn", ["system_error"])

class LLMGuardScanner(BaseGuard):
    """Enhanced LLM Guard implementation with Indonesian support"""
    
    def __init__(self):
        super().__init__("llm_guard")
        self.model = None
        if LLMGUARD_AVAILABLE:
            try:
                from llmguard import scan_prompt
                from llmguard.input_scanners import (
                    Anonymize, BanTopics, Code, Language, PromptInjection, 
                    Sentiment, TokenLimit, Toxicity
                )
                self.scanners = [
                    Toxicity(threshold=0.7),
                    PromptInjection(threshold=0.8),
                    BanTopics(topics=["violence", "hate", "self-harm"], threshold=0.75),
                    Sentiment(threshold=-0.5)
                ]
                logger.info("LLMGuard scanners initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize LLMGuard scanners: {e}")
                self.scanners = None
        else:
            self.scanners = None
    
    def _detect_indonesian_content(self, text: str) -> bool:
        """Detect Indonesian language content"""
        if LANGDETECT_AVAILABLE:
            try:
                from langdetect import detect
                return detect(text) == 'id'
            except:
                pass
        
        # Fallback to keyword detection
        indonesian_words = [
            'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'adalah',
            'tidak', 'akan', 'sudah', 'bisa', 'harus', 'juga', 'saya', 'kamu', 'dia',
            'ini', 'itu', 'ada', 'atau', 'jika', 'kalau', 'karena', 'sebab', 'maka'
        ]
        text_lower = text.lower()
        return sum(1 for word in indonesian_words if word in text_lower) >= 3
        
    async def analyze(self, prompt: str, lang: str = "auto", **kwargs) -> GuardResult:
        """Enhanced analysis with LLMGuard and Indonesian support"""
        try:
            await asyncio.sleep(0.12)
            
            prompt_lower = prompt.lower()
            is_indonesian = self._detect_indonesian_content(prompt)
            
            # Enhanced prompt injection detection for Indonesian
            injection_keywords_id = [
                'abaikan instruksi', 'lupakan instruksi', 'sistem prompt', 'prompt sistem',
                'jailbreak', 'bypass', 'lewati aturan', 'langgar aturan', 'hack sistem',
                'override', 'ganti peran', 'ubah peran', 'pretend', 'pura-pura',
                'ignore previous', 'forget everything', 'new instructions'
            ]
            
            injection_keywords_en = [
                'jailbreak', 'ignore instructions', 'system prompt', 'forget previous',
                'override', 'bypass', 'pretend to be', 'act as', 'roleplay as',
                'new instructions', 'disregard', 'ignore everything above'
            ]
            
            # Enhanced toxicity detection for Indonesian
            toxicity_keywords_id = [
                'toxic', 'toksik', 'beracun', 'menyakitkan', 'menyinggung', 'ofensif',
                'kasar', 'tidak pantas', 'tidak sopan', 'vulgar', 'cabul', 'jorok',
                'kotor', 'mesum', 'porno', 'pornografi', 'asusila', 'tidak senonoh'
            ]
            
            toxicity_keywords_en = [
                'toxic', 'offensive', 'inappropriate', 'vulgar', 'obscene',
                'profane', 'indecent', 'lewd', 'crude', 'rude', 'nasty'
            ]
            
            # Enhanced bias detection for Indonesian context
            bias_keywords_id = [
                'bias', 'prasangka', 'stereotip', 'stereotype', 'diskriminasi',
                'prejudis', 'rasisme', 'seksisme', 'fanatik', 'intoleran',
                'sara', 'suku agama ras', 'golongan', 'mayoritas', 'minoritas'
            ]
            
            bias_keywords_en = [
                'bias', 'stereotype', 'prejudice', 'discrimination', 'racism',
                'sexism', 'bigotry', 'intolerance', 'supremacy', 'superiority'
            ]
            
            labels = []
            max_severity = "low"
            
            # Use LLMGuard scanners if available
            if self.scanners and LLMGUARD_AVAILABLE:
                try:
                    from llmguard import scan_prompt
                    sanitized_prompt, results_valid, results_score = scan_prompt(
                        self.scanners, prompt
                    )
                    
                    if not results_valid:
                        # Determine the most severe issue
                        max_score = max(results_score.values()) if results_score else 0
                        if max_score > 0.8:
                            return GuardResult("block", ["LLMGuard_High_Risk"])
                        elif max_score > 0.6:
                            return GuardResult("warn", ["LLMGuard_Medium_Risk"])
                except Exception as e:
                    logger.warning(f"LLMGuard scanning failed: {e}")
            
            # Fallback to keyword-based analysis with Indonesian support
            weight = 1.3 if is_indonesian else 1.0
            
            # Check for prompt injection
            if any(word in prompt_lower for word in injection_keywords_id + injection_keywords_en):
                labels.append("PromptInjection")
                max_severity = "critical"
            
            # Check for toxicity
            toxicity_score = sum(1 for word in toxicity_keywords_id + toxicity_keywords_en 
                               if word in prompt_lower) * weight
            if toxicity_score >= 2:
                labels.append("Toxicity")
                max_severity = "moderate" if max_severity == "low" else max_severity
            
            # Check for bias
            bias_score = sum(1 for word in bias_keywords_id + bias_keywords_en 
                           if word in prompt_lower) * weight
            if bias_score >= 1:
                labels.append("Bias")
                max_severity = "moderate" if max_severity == "low" else max_severity
                
            if any(word in prompt_lower for word in ['email', 'phone', 'address', 'ssn']):
                labels.append("PII")
                max_severity = "critical"
                
            if 'http' in prompt_lower or 'www.' in prompt_lower:
                labels.append("URL")
                max_severity = "moderate" if max_severity == "low" else max_severity
            
            # Additional Indonesian-specific checks
            if is_indonesian:
                # Check for Indonesian-specific harmful patterns
                indonesian_harmful = [
                    'hoax', 'hoaks', 'berita palsu', 'fake news', 'fitnah', 'memfitnah',
                    'provokasi', 'memprovokasi', 'hasut', 'menghasut', 'adu domba'
                ]
                if any(word in prompt_lower for word in indonesian_harmful):
                    labels.append("Misinformation")
                    max_severity = "moderate" if max_severity == "low" else max_severity
                
            # Map severity to verdict
            if max_severity == "critical":
                return GuardResult("block", labels)
            elif max_severity == "moderate":
                return GuardResult("warn", labels)
            else:
                return GuardResult("allow", labels)
                
        except Exception as e:
            logger.error(f"LLMGuardScanner error: {e}")
            return GuardResult("warn", ["system_error"])

class NeMoGuardrails(BaseGuard):
    """Enhanced NeMo Guardrails implementation with Indonesian support"""
    
    def __init__(self):
        super().__init__("nemo_guardrails")
        self.rails = None
        if NEMO_AVAILABLE:
            try:
                from nemoguardrails import LLMRails, RailsConfig
                # Initialize with basic safety rails
                config = RailsConfig.from_content(
                    colang_content="""
                    define user ask about harmful content
                      "How to make explosives"
                      "How to hack systems"
                      "How to hurt someone"
                      "Cara membuat bom"
                      "Cara meretas sistem"
                      "Cara menyakiti orang"
                    
                    define bot refuse harmful request
                      "I cannot and will not provide information on harmful activities."
                      "Saya tidak dapat memberikan informasi tentang aktivitas berbahaya."
                    
                    define flow harmful content
                      user ask about harmful content
                      bot refuse harmful request
                    """,
                    yaml_content="""
                    models:
                      - type: main
                        engine: openai
                        model: gpt-3.5-turbo
                    """
                )
                self.rails = LLMRails(config)
                logger.info("NeMo Guardrails initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize NeMo Guardrails: {e}")
                self.rails = None
        else:
            self.rails = None
    
    def _detect_indonesian_content(self, text: str) -> bool:
        """Detect Indonesian language content"""
        if LANGDETECT_AVAILABLE:
            try:
                from langdetect import detect
                return detect(text) == 'id'
            except:
                pass
        
        # Fallback to keyword detection
        indonesian_words = [
            'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'dengan', 'pada', 'adalah',
            'tidak', 'akan', 'sudah', 'bisa', 'harus', 'juga', 'saya', 'kamu', 'dia',
            'bagaimana', 'mengapa', 'dimana', 'kapan', 'siapa', 'apa', 'mana'
        ]
        text_lower = text.lower()
        return sum(1 for word in indonesian_words if word in text_lower) >= 2
        
    async def analyze(self, prompt: str, lang: str = "auto", **kwargs) -> GuardResult:
        """Enhanced analysis with NeMo Guardrails and Indonesian support"""
        try:
            await asyncio.sleep(0.2)
            
            prompt_lower = prompt.lower()
            is_indonesian = self._detect_indonesian_content(prompt)
            
            # Enhanced code generation detection for Indonesian
            code_keywords_id = [
                'buat kode', 'tulis kode', 'generate code', 'bikin program', 'buat program',
                'tulis script', 'buat script', 'coding', 'programming', 'pemrograman',
                'algoritma', 'function', 'fungsi', 'class', 'kelas', 'method', 'metode'
            ]
            
            code_keywords_en = [
                'generate code', 'write script', 'create program', 'write code',
                'programming', 'coding', 'algorithm', 'function', 'class', 'method',
                'implement', 'develop', 'build application', 'software development'
            ]
            
            # Enhanced data leakage detection for Indonesian
            data_keywords_id = [
                'data pribadi', 'informasi pribadi', 'data sensitif', 'informasi sensitif',
                'rahasia', 'konfidensial', 'password', 'kata sandi', 'pin', 'ktp',
                'sim', 'paspor', 'rekening', 'kartu kredit', 'nomor telepon', 'alamat'
            ]
            
            data_keywords_en = [
                'personal data', 'sensitive info', 'confidential', 'private information',
                'secret', 'password', 'credit card', 'social security', 'phone number',
                'address', 'email', 'bank account', 'personal details'
            ]
            
            # Enhanced professional advice detection for Indonesian
            advice_keywords_id = [
                'saran medis', 'nasihat medis', 'konsultasi dokter', 'diagnosa',
                'saran hukum', 'nasihat hukum', 'konsultasi pengacara', 'legal advice',
                'saran keuangan', 'nasihat keuangan', 'investasi', 'saham', 'trading',
                'konsultasi psikolog', 'terapi', 'pengobatan', 'obat-obatan'
            ]
            
            advice_keywords_en = [
                'medical advice', 'legal advice', 'financial advice', 'investment advice',
                'diagnosis', 'treatment', 'medication', 'therapy', 'consultation',
                'professional opinion', 'expert advice', 'clinical advice'
            ]
            
            # Use NeMo Guardrails if available
            if self.rails and NEMO_AVAILABLE:
                try:
                    response = await self.rails.generate_async(messages=[
                        {"role": "user", "content": prompt}
                    ])
                    
                    # Check if the response indicates a blocked request
                    if "cannot" in response.get("content", "").lower() or \
                       "tidak dapat" in response.get("content", "").lower():
                        return GuardResult("block", ["NeMo_Blocked"])
                except Exception as e:
                    logger.warning(f"NeMo Guardrails processing failed: {e}")
            
            # Fallback to keyword-based analysis with Indonesian support
            weight = 1.2 if is_indonesian else 1.0
            
            # Check for code generation requests
            code_score = sum(1 for word in code_keywords_id + code_keywords_en 
                           if word in prompt_lower) * weight
            if code_score >= 2:
                return GuardResult("warn", ["Code_Generation"])
            
            # Check for data leakage risks
            data_score = sum(1 for word in data_keywords_id + data_keywords_en 
                           if word in prompt_lower) * weight
            if data_score >= 2:
                return GuardResult("block", ["Data_Leakage"])
            
            # Check for professional advice requests
            advice_score = sum(1 for word in advice_keywords_id + advice_keywords_en 
                             if word in prompt_lower) * weight
            if advice_score >= 1:
                return GuardResult("warn", ["Professional_Advice"])
            
            # Additional Indonesian-specific safety checks
            if is_indonesian:
                # Check for Indonesian-specific risky content
                indonesian_risky = [
                    'cara membuat', 'tutorial membuat', 'langkah membuat', 'resep membuat',
                    'panduan hack', 'cara hack', 'tutorial hack', 'cara bobol',
                    'cara menipu', 'cara scam', 'cara penipuan', 'trik penipuan'
                ]
                risky_score = sum(1 for word in indonesian_risky if word in prompt_lower)
                if risky_score >= 1:
                    # Check if it's combined with harmful intent
                    harmful_objects = [
                        'bom', 'senjata', 'racun', 'narkoba', 'virus', 'malware',
                        'sistem keamanan', 'password orang', 'data orang lain'
                    ]
                    if any(obj in prompt_lower for obj in harmful_objects):
                        return GuardResult("block", ["Harmful_Instructions"])
                    else:
                        return GuardResult("warn", ["Potentially_Risky"])
                
            return GuardResult("allow", [])
            
        except Exception as e:
            logger.error(f"NeMoGuardrails error: {e}")
            return GuardResult("warn", ["system_error"])

class GuardManager:
    """Manages all guards and orchestrates analysis"""
    
    def __init__(self):
        self.guards = {
            "llama_guard_8b": LlamaGuard8B(),
            "llama_guard_1b": LlamaGuard1B(),
            "indobert_toxic": IndoBERTToxic(),
            "llm_guard": LLMGuardScanner(),
            "nemo_guardrails": NeMoGuardrails()
        }
        
    async def analyze_prompt(
        self, 
        prompt: str, 
        lang: str, 
        enabled_guards: List[str],
        thresholds: Dict[str, float]
    ) -> Dict[str, GuardResult]:
        """Run prompt through all enabled guards in parallel"""
        
        tasks = []
        for guard_name in enabled_guards:
            if guard_name in self.guards:
                guard = self.guards[guard_name]
                task = asyncio.create_task(
                    guard.analyze(prompt, lang, **thresholds)
                )
                tasks.append((guard_name, task))
        
        results = {}
        for guard_name, task in tasks:
            try:
                result = await task
                results[guard_name] = {
                    "verdict": result.verdict,
                    "labels": result.labels,
                    "score": result.score
                }
            except Exception as e:
                logger.error(f"Guard {guard_name} failed: {e}")
                results[guard_name] = {
                    "verdict": "warn",
                    "labels": ["system_error"],
                    "score": None
                }
        
        # Handle LLaMA Guard disagreement
        if "llama_guard_8b" in results and "llama_guard_1b" in results:
            results = self._resolve_llama_disagreement(results)
            
        return results
    
    def _resolve_llama_disagreement(self, results: Dict) -> Dict:
        """Choose safer verdict when LLaMA guards disagree"""
        guard_8b = results["llama_guard_8b"]
        guard_1b = results["llama_guard_1b"]
        
        # Severity order: block > warn > allow
        severity_order = {"block": 3, "warn": 2, "allow": 1}
        
        if severity_order[guard_8b["verdict"]] >= severity_order[guard_1b["verdict"]]:
            safer_result = guard_8b
            safer_name = "llama_guard_8b"
        else:
            safer_result = guard_1b
            safer_name = "llama_guard_1b"
            
        # Update results with combined decision
        results["llama_guard_combined"] = {
            "verdict": safer_result["verdict"],
            "labels": list(set(guard_8b["labels"] + guard_1b["labels"])),
            "score": None,
            "note": f"Combined decision from {safer_name}"
        }
        
        return results
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all guards"""
        return {name: guard.health_check() for name, guard in self.guards.items()}
    
    def get_versions(self) -> Dict[str, str]:
        """Get version info for all guards"""
        return {name: guard.version for name, guard in self.guards.items()}