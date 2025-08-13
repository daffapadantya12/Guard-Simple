#!/usr/bin/env python3
"""
Quick test script to verify API key authentication is working
"""

import requests
import os

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "dffhnfpdnty0194392429340")

# Headers with API key
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def test_protected_endpoint():
    """Test a protected endpoint with API key"""
    print("Testing /config endpoint with API key...")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/config",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ SUCCESS: API key authentication working!")
            config = response.json()
            print(f"   Enable all: {config.get('enable_all')}")
            print(f"   Guards: {list(config.get('guards', {}).keys())}")
        else:
            print(f"❌ ERROR {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def test_unprotected_endpoint():
    """Test an unprotected endpoint"""
    print("\nTesting /healthz endpoint (no auth required)...")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/healthz",
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ SUCCESS: Health endpoint accessible without auth")
            health = response.json()
            print(f"   Status: {health.get('status')}")
        else:
            print(f"❌ ERROR {response.status_code}: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

def test_unauthorized_access():
    """Test access without API key"""
    print("\nTesting /config endpoint without API key...")
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/config",
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 403:
            print("✅ SUCCESS: Unauthorized access properly blocked (403)")
        elif response.status_code == 401:
            print("✅ SUCCESS: Unauthorized access properly blocked (401)")
        else:
            print(f"❌ UNEXPECTED: Got {response.status_code}, expected 401/403")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    print("=== API Key Authentication Test ===")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"API Key: {API_KEY[:10]}...")
    print()
    
    test_protected_endpoint()
    test_unprotected_endpoint()
    test_unauthorized_access()
    
    print("\n=== Test Complete ===")