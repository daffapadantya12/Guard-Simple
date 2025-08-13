#!/usr/bin/env python3
"""
Test script for Guard API with API Key Authentication

This script demonstrates how to use the Guard API with the required API key authentication.
"""

import requests
import json
import os
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "dffhnfpdnty0194392429340")  # Updated default key

# Headers with API key
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def test_analyze_endpoint():
    """Test the /analyze endpoint with API key authentication"""
    print("\n=== Testing /analyze endpoint ===")
    
    # Test data
    test_prompts = [
        "Hello, how are you today?",
        "This is a test prompt for safety analysis",
        "Can you help me with my homework?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: Analyzing prompt: '{prompt[:50]}...'")
        
        payload = {
            "prompt": prompt,
            "user_id": f"test_user_{i}",
            "lang": "auto"
        }
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/analyze",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success: Final verdict = {result['final']}")
                print(f"   Guards analyzed: {list(result['per_guard'].keys())}")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")

def test_config_endpoint():
    """Test the /config endpoint with API key authentication"""
    print("\n=== Testing /config endpoint ===")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/config",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            config = response.json()
            print("✅ Config retrieved successfully:")
            print(f"   Enable all: {config.get('enable_all')}")
            print(f"   Active guards: {list(config.get('guards', {}).keys())}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def test_metrics_endpoint():
    """Test the /metrics endpoint with API key authentication"""
    print("\n=== Testing /metrics endpoint ===")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/metrics",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            metrics = response.json()
            print("✅ Metrics retrieved successfully:")
            print(f"   Total requests: {metrics.get('total_requests', 0)}")
            print(f"   Cache hits: {metrics.get('cache_hits', 0)}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def test_health_endpoint():
    """Test the /healthz endpoint (should work without API key)"""
    print("\n=== Testing /healthz endpoint (no auth required) ===")
    
    try:
        # Test without API key
        response = requests.get(
            f"{API_BASE_URL}/healthz",
            timeout=10
        )
        
        if response.status_code == 200:
            health = response.json()
            print("✅ Health check successful (no auth required):")
            print(f"   Status: {health.get('status')}")
            print(f"   Redis: {health.get('redis')}")
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def test_unauthorized_access():
    """Test API access without proper authentication"""
    print("\n=== Testing unauthorized access ===")
    
    # Test with wrong API key
    wrong_headers = {
        "Authorization": "Bearer wrong-api-key",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/config",
            headers=wrong_headers,
            timeout=10
        )
        
        if response.status_code == 401:
            print("✅ Unauthorized access properly blocked (401)")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Expected 401, got {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    
    # Test without API key
    try:
        response = requests.get(
            f"{API_BASE_URL}/config",
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 403:
            print("✅ Missing API key properly blocked (403)")
        else:
            print(f"❌ Expected 403, got {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def main():
    """Run all API tests"""
    print("🔐 Guard API Authentication Test Suite")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Using API Key: {API_KEY[:10]}...{API_KEY[-4:]}")
    
    # Test health endpoint first (no auth required)
    test_health_endpoint()
    
    # Test unauthorized access
    test_unauthorized_access()
    
    # Test authenticated endpoints
    test_config_endpoint()
    test_metrics_endpoint()
    test_analyze_endpoint()
    
    print("\n🎉 API authentication tests completed!")
    print("\n📝 Usage Notes:")
    print("- Set API_KEY environment variable for custom key")
    print("- Include 'Authorization: Bearer <your-api-key>' header in requests")
    print("- Health endpoint (/healthz) doesn't require authentication")
    print("- All other endpoints require valid API key")

if __name__ == "__main__":
    main()