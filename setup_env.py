#!/usr/bin/env python3
"""
Environment Setup Script for GUARD-SIMPLE

This script helps you configure the .env file with appropriate settings
for your hardware and preferences.
"""

import os
import sys

def detect_hardware():
    """Detect available hardware"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return {
                'cuda_available': True,
                'gpu_name': gpu_name,
                'gpu_memory_gb': gpu_memory
            }
    except ImportError:
        pass
    
    return {'cuda_available': False}

def get_user_preferences():
    """Get user preferences for configuration"""
    print("\n=== GUARD-SIMPLE Environment Setup ===")
    
    # Detect hardware
    hardware = detect_hardware()
    
    if hardware['cuda_available']:
        print(f"\n✅ CUDA GPU detected: {hardware['gpu_name']}")
        print(f"   GPU Memory: {hardware['gpu_memory_gb']:.1f} GB")
        
        device_choice = input("\nChoose device (cuda/cpu/auto) [auto]: ").strip().lower()
        if device_choice not in ['cuda', 'cpu', 'auto', '']:
            device_choice = 'auto'
        elif device_choice == '':
            device_choice = 'auto'
    else:
        print("\n❌ No CUDA GPU detected")
        device_choice = 'cpu'
    
    # Quantization preferences
    if device_choice in ['cuda', 'auto'] and hardware['cuda_available']:
        print("\n🔧 Quantization can reduce memory usage and improve speed")
        use_quantization = input("Enable quantization? (y/n) [y]: ").strip().lower()
        use_quantization = use_quantization != 'n'
        
        if use_quantization:
            print("\nQuantization options:")
            print("  4-bit: Lower memory, faster (recommended)")
            print("  8-bit: Higher quality, more memory")
            bits = input("Choose quantization bits (4/8) [4]: ").strip()
            bits = '8' if bits == '8' else '4'
        else:
            bits = '4'
    else:
        use_quantization = False
        bits = '4'
    
    # Hugging Face token
    print("\n🔑 Hugging Face Token (required for LLaMA Guard models)")
    print("   Get your token from: https://huggingface.co/settings/tokens")
    hf_token = input("Enter your Hugging Face token (or press Enter to skip): ").strip()
    
    return {
        'device': device_choice,
        'use_quantization': use_quantization,
        'quantization_bits': bits,
        'hf_token': hf_token
    }

def create_env_file(preferences):
    """Create .env file with user preferences"""
    env_content = f"""# Hugging Face Configuration
HUGGINGFACE_TOKEN={preferences['hf_token']}

# Device Configuration
# Options: auto, cuda, cpu
DEVICE={preferences['device']}

# Model Quantization Configuration
# Enable quantization using BitsAndBytes (reduces memory usage)
USE_QUANTIZATION={str(preferences['use_quantization']).lower()}

# Quantization Settings
QUANTIZATION_BITS={preferences['quantization_bits']}
QUANTIZATION_TYPE=nf4
USE_DOUBLE_QUANT=true
COMPUTE_DTYPE=float16

# Model Loading Configuration
TRUST_REMOTE_CODE=true
TORCH_DTYPE=float16

# Cache Configuration
HF_CACHE_DIR=./models_cache
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("\n✅ .env file created successfully!")
    print("\n📋 Configuration Summary:")
    print(f"   Device: {preferences['device']}")
    print(f"   Quantization: {'Enabled' if preferences['use_quantization'] else 'Disabled'}")
    if preferences['use_quantization']:
        print(f"   Quantization bits: {preferences['quantization_bits']}")
    print(f"   HF Token: {'Set' if preferences['hf_token'] else 'Not set'}")
    
    if not preferences['hf_token']:
        print("\n⚠️  Warning: No Hugging Face token provided.")
        print("   LLaMA Guard models require authentication.")
        print("   You can add your token to the .env file later.")

def main():
    """Main setup function"""
    if os.path.exists('.env'):
        overwrite = input(".env file already exists. Overwrite? (y/n) [n]: ").strip().lower()
        if overwrite != 'y':
            print("Setup cancelled.")
            return
    
    try:
        preferences = get_user_preferences()
        create_env_file(preferences)
        
        print("\n🚀 Setup complete! You can now run the application.")
        print("\n💡 Tips:")
        print("   - Edit .env file to modify settings")
        print("   - Restart the application after changing .env")
        print("   - Check logs for model loading status")
        
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()