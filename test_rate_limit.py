import requests
import time
import json

def test_rate_limit():
    url = "http://localhost:8000/analyze"
    headers = {"Content-Type": "application/json"}
    
    # Test payload
    payload = {
        "prompt": "Hello, this is a test message",
        "user_id": "test_user",
        "lang": "auto"
    }
    
    print("Testing rate limit (50 requests per 60 seconds)...\n")
    
    success_count = 0
    rate_limited_count = 0
    error_count = 0
    
    # Make 55 requests quickly to trigger rate limiting
    for i in range(1, 56):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                success_count += 1
                result = response.json()
                print(f"Request {i:2d}: ✅ SUCCESS (200) - Final verdict: {result.get('final', 'unknown')}")
            elif response.status_code == 429:
                rate_limited_count += 1
                print(f"Request {i:2d}: 🚫 RATE LIMITED (429) - Too Many Requests")
            else:
                error_count += 1
                print(f"Request {i:2d}: ❌ ERROR ({response.status_code}) - {response.text[:100]}")
                
        except requests.exceptions.ConnectionError:
            error_count += 1
            print(f"Request {i:2d}: ❌ CONNECTION ERROR - Server not running?")
        except requests.exceptions.Timeout:
            error_count += 1
            print(f"Request {i:2d}: ❌ TIMEOUT ERROR - Request took too long")
        except requests.exceptions.RequestException as e:
            error_count += 1
            print(f"Request {i:2d}: ❌ REQUEST ERROR - {str(e)[:100]}")
        except Exception as e:
            error_count += 1
            print(f"Request {i:2d}: ❌ UNEXPECTED ERROR - {str(e)[:100]}")
        
        # Small delay between requests
        time.sleep(0.2)
    
    print(f"\n📊 Test Results:")
    print(f"✅ Successful requests: {success_count}")
    print(f"🚫 Rate limited requests: {rate_limited_count}")
    print(f"❌ Error requests: {error_count}")
    print(f"📈 Total requests: {success_count + rate_limited_count + error_count}")
    
    print(f"\n🔍 Analysis:")
    if error_count > 0:
        print("⚠️  Some requests failed due to connection/server errors.")
        print("   Make sure the server is running with: python main.py")
    
    if success_count <= 50 and rate_limited_count > 0:
        print("🎉 Rate limiting is working correctly!")
        print(f"   - Allowed {success_count} requests (≤50 expected)")
        print(f"   - Blocked {rate_limited_count} requests with 429 status")
    elif success_count > 50:
        print("⚠️  Rate limiting may not be working - too many successful requests.")
    elif rate_limited_count == 0 and success_count > 0:
        print("⚠️  Rate limiting may not be active - no 429 responses received.")
    else:
        print("❓ Unable to determine rate limiting status due to errors.")

if __name__ == "__main__":
    print("🧪 FastAPI Rate Limit Test")
    print("=" * 50)
    test_rate_limit()
    print("\n💡 Note: Rate limit resets after 60 seconds.")
    print("   You can run this test again after waiting 1 minute.")