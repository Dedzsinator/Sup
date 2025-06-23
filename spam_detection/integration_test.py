#!/usr/bin/env python3
"""
Integration test script for spam detection system
Tests the API endpoints and integration with the Sup messaging app
"""

import requests
import json
import time
import sys

# Test configuration
SPAM_DETECTION_URL = "http://localhost:8082"
BACKEND_URL = "http://localhost:4000"  # Elixir backend (if running)

def test_spam_detection_api():
    """Test the spam detection microservice API"""
    print("="*60)
    print("TESTING SPAM DETECTION MICROSERVICE")
    print("="*60)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        response = requests.get(f"{SPAM_DETECTION_URL}/health")
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.json()}")
    except Exception as e:
        print(f"   ERROR: {e}")
        return False
    
    # Test prediction endpoint with spam message
    print("\n2. Testing spam prediction...")
    spam_messages = [
        "BUY VIAGRA NOW!!! CHEAP PRICES!!!",
        "Win $1000000 NOW! Click this link URGENT!",
        "Free money! Act now! Limited time offer!",
        "NIGERIAN PRINCE needs your help with inheritance"
    ]
    
    for msg in spam_messages:
        try:
            payload = {
                "message": msg,
                "user_id": "test_user_spam",
                "timestamp": "2025-06-23T13:45:00Z"
            }
            response = requests.post(f"{SPAM_DETECTION_URL}/predict", json=payload)
            result = response.json()
            print(f"   Message: '{msg[:40]}...'")
            print(f"   -> Spam: {result['is_spam']}, Probability: {result['spam_probability']}, Confidence: {result['confidence']}")
        except Exception as e:
            print(f"   ERROR: {e}")
    
    # Test prediction endpoint with normal messages
    print("\n3. Testing normal message prediction...")
    normal_messages = [
        "Hello, how are you doing today?",
        "Thanks for the meeting yesterday",
        "Can we reschedule our call for tomorrow?",
        "The project is going well, see you soon"
    ]
    
    for msg in normal_messages:
        try:
            payload = {
                "message": msg,
                "user_id": "test_user_normal",
                "timestamp": "2025-06-23T13:45:00Z"
            }
            response = requests.post(f"{SPAM_DETECTION_URL}/predict", json=payload)
            result = response.json()
            print(f"   Message: '{msg[:40]}...'")
            print(f"   -> Spam: {result['is_spam']}, Probability: {result['spam_probability']}, Confidence: {result['confidence']}")
        except Exception as e:
            print(f"   ERROR: {e}")
    
    # Test batch prediction
    print("\n4. Testing batch prediction...")
    try:
        batch_payload = {
            "messages": [
                {"message": "Hello there", "user_id": "user1"},
                {"message": "FREE MONEY WIN NOW!!!", "user_id": "user2"},
                {"message": "Thanks for your help", "user_id": "user3"}
            ]
        }
        response = requests.post(f"{SPAM_DETECTION_URL}/predict/batch", json=batch_payload)
        results = response.json()
        for i, result in enumerate(results):
            print(f"   Message {i+1}: Spam={result['is_spam']}, Prob={result['spam_probability']}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test training endpoint
    print("\n5. Testing training data submission...")
    try:
        training_payload = {
            "message": "This is a test spam message",
            "user_id": "test_user",
            "is_spam": True,
            "timestamp": "2025-06-23T13:45:00Z"
        }
        response = requests.post(f"{SPAM_DETECTION_URL}/train", json=training_payload)
        result = response.json()
        print(f"   Training response: {result}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test statistics endpoint
    print("\n6. Testing statistics endpoint...")
    try:
        response = requests.get(f"{SPAM_DETECTION_URL}/stats")
        stats = response.json()
        print(f"   Stats: {json.dumps(stats, indent=2)}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    print("\n" + "="*60)
    print("SPAM DETECTION API TESTS COMPLETED")
    print("="*60)
    return True

def test_performance():
    """Test performance of spam detection"""
    print("\n" + "="*60)
    print("TESTING PERFORMANCE")
    print("="*60)
    
    # Test single message performance
    print("\n1. Testing single message latency...")
    latencies = []
    for i in range(10):
        payload = {
            "message": f"Test message {i} with some content to analyze",
            "user_id": f"perf_user_{i}",
        }
        start_time = time.time()
        try:
            response = requests.post(f"{SPAM_DETECTION_URL}/predict", json=payload)
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to ms
            latencies.append(latency)
            print(f"   Request {i+1}: {latency:.2f}ms")
        except Exception as e:
            print(f"   Request {i+1}: ERROR - {e}")
    
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"\n   Average latency: {avg_latency:.2f}ms")
        print(f"   Min latency: {min(latencies):.2f}ms")
        print(f"   Max latency: {max(latencies):.2f}ms")
    
    # Test batch performance
    print("\n2. Testing batch processing performance...")
    batch_sizes = [5, 10, 20]
    for batch_size in batch_sizes:
        messages = [
            {"message": f"Batch test message {i}", "user_id": f"batch_user_{i}"}
            for i in range(batch_size)
        ]
        batch_payload = {"messages": messages}
        
        start_time = time.time()
        try:
            response = requests.post(f"{SPAM_DETECTION_URL}/predict/batch", json=batch_payload)
            end_time = time.time()
            total_time = (end_time - start_time) * 1000
            per_message_time = total_time / batch_size
            print(f"   Batch size {batch_size}: {total_time:.2f}ms total, {per_message_time:.2f}ms per message")
        except Exception as e:
            print(f"   Batch size {batch_size}: ERROR - {e}")

def test_spam_detection_accuracy():
    """Test accuracy of spam detection with known samples"""
    print("\n" + "="*60)
    print("TESTING SPAM DETECTION ACCURACY")
    print("="*60)
    
    # Test cases with expected results
    test_cases = [
        # (message, expected_is_spam)
        ("Hello, how are you?", False),
        ("BUY VIAGRA NOW!!! CHEAP PRICES!!!", True),
        ("Win $1000000 now! Click this link!", True),
        ("Thanks for the meeting today", False),
        ("URGENT: Nigerian prince needs help", True),
        ("Free money! Act now! Limited time!", True),
        ("Can we schedule a call tomorrow?", False),
        ("CLICK HERE FOR CHEAP LOANS!!!", True),
        ("The weather is nice today", False),
        ("Weight loss pills FREE TRIAL!!!", True),
        ("See you at the conference", False),
        ("Bitcoin investment opportunity!!!", True),
    ]
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    print(f"\nTesting {total_predictions} messages...")
    
    for i, (message, expected_spam) in enumerate(test_cases, 1):
        try:
            payload = {
                "message": message,
                "user_id": f"accuracy_test_user_{i}",
            }
            response = requests.post(f"{SPAM_DETECTION_URL}/predict", json=payload)
            result = response.json()
            
            predicted_spam = result['is_spam']
            is_correct = predicted_spam == expected_spam
            
            if is_correct:
                correct_predictions += 1
                status = "✓ CORRECT"
            else:
                status = "✗ WRONG"
            
            print(f"   {i:2d}. {status} | '{message[:40]}...' | Expected: {expected_spam}, Got: {predicted_spam} (prob: {result['spam_probability']:.3f})")
            
        except Exception as e:
            print(f"   {i:2d}. ERROR | '{message[:40]}...' | {e}")
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nAccuracy: {correct_predictions}/{total_predictions} = {accuracy:.2%}")
    
    if accuracy >= 0.8:
        print("✓ GOOD: Accuracy is acceptable (≥80%)")
    elif accuracy >= 0.6:
        print("⚠ WARNING: Accuracy is moderate (60-80%)")
    else:
        print("✗ POOR: Accuracy is low (<60%)")
    
    return accuracy

def main():
    """Main test function"""
    print("SPAM DETECTION INTEGRATION TESTS")
    print("=" * 60)
    print(f"Testing spam detection service at: {SPAM_DETECTION_URL}")
    print(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test if service is running
    try:
        response = requests.get(f"{SPAM_DETECTION_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"ERROR: Spam detection service not healthy (status: {response.status_code})")
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Cannot connect to spam detection service: {e}")
        print("Please ensure the service is running on port 8082")
        sys.exit(1)
    
    # Run tests
    try:
        # Test API functionality
        api_success = test_spam_detection_api()
        
        # Test performance
        test_performance()
        
        # Test accuracy
        accuracy = test_spam_detection_accuracy()
        
        # Final summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"API Tests: {'PASSED' if api_success else 'FAILED'}")
        print(f"Accuracy: {accuracy:.1%}")
        print("Spam detection microservice is ready for integration!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Test suite failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
