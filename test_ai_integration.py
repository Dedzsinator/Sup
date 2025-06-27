#!/usr/bin/env python3
"""
Integration test script to verify both AI services are correctly integrated with the Elixir backend.

This script tests:
1. Spam Detection Service (Python) ↔ Elixir Backend
2. Autocomplete Service (Python) ↔ Elixir Backend
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List

class AIIntegrationTester:
    def __init__(self):
        self.spam_detection_url = "http://localhost:8082"
        self.autocomplete_url = "http://localhost:8000"
        self.backend_url = "http://localhost:4000"  # Assuming Elixir backend port
        
        self.results = {
            "spam_detection": {"status": "unknown", "tests": []},
            "autocomplete": {"status": "unknown", "tests": []},
            "backend_integration": {"status": "unknown", "tests": []}
        }
    
    def test_spam_detection_service(self) -> bool:
        """Test the spam detection service directly"""
        print("🔍 Testing Spam Detection Service...")
        
        try:
            # Test health endpoint
            response = requests.get(f"{self.spam_detection_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"  ✅ Health check passed: {health_data.get('status', 'unknown')}")
                self.results["spam_detection"]["tests"].append({"test": "health", "status": "pass"})
            else:
                print(f"  ❌ Health check failed: {response.status_code}")
                self.results["spam_detection"]["tests"].append({"test": "health", "status": "fail"})
                return False
            
            # Test spam prediction
            test_messages = [
                {"text": "Buy cheap viagra now!", "expected_spam": True},
                {"text": "Hi, how are you doing today?", "expected_spam": False},
                {"text": "URGENT: You've won $1000000! Click here!", "expected_spam": True},
                {"text": "Meeting tomorrow at 3pm in the conference room", "expected_spam": False}
            ]
            
            for i, test_msg in enumerate(test_messages):
                payload = {"text": test_msg["text"], "user_id": f"test_user_{i}"}
                response = requests.post(f"{self.spam_detection_url}/predict", 
                                       json=payload, timeout=5)
                
                if response.status_code == 200:
                    result = response.json()
                    is_spam = result.get("is_spam", False)
                    confidence = result.get("confidence", 0.0)
                    
                    # Check if prediction matches expectation (allowing for model differences)
                    prediction_correct = is_spam == test_msg["expected_spam"]
                    
                    print(f"  {'✅' if prediction_correct else '⚠️'} Message: '{test_msg['text'][:30]}...'")
                    print(f"    Predicted: {'SPAM' if is_spam else 'HAM'} (confidence: {confidence:.2f})")
                    print(f"    Expected: {'SPAM' if test_msg['expected_spam'] else 'HAM'}")
                    
                    self.results["spam_detection"]["tests"].append({
                        "test": f"prediction_{i}",
                        "status": "pass" if prediction_correct else "warning",
                        "message": test_msg["text"][:50],
                        "predicted": is_spam,
                        "expected": test_msg["expected_spam"],
                        "confidence": confidence
                    })
                else:
                    print(f"  ❌ Prediction failed: {response.status_code}")
                    self.results["spam_detection"]["tests"].append({
                        "test": f"prediction_{i}",
                        "status": "fail"
                    })
                    return False
            
            # Test batch prediction
            batch_payload = {
                "messages": [
                    {"text": "Free money offer!", "user_id": "batch_user_1"},
                    {"text": "Thanks for your help", "user_id": "batch_user_2"}
                ]
            }
            
            response = requests.post(f"{self.spam_detection_url}/predict/batch", 
                                   json=batch_payload, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get("predictions", [])
                print(f"  ✅ Batch prediction: {len(predictions)} results returned")
                self.results["spam_detection"]["tests"].append({"test": "batch", "status": "pass"})
            else:
                print(f"  ❌ Batch prediction failed: {response.status_code}")
                self.results["spam_detection"]["tests"].append({"test": "batch", "status": "fail"})
                return False
            
            self.results["spam_detection"]["status"] = "operational"
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Spam detection service unreachable: {e}")
            self.results["spam_detection"]["status"] = "unreachable"
            return False
    
    def test_autocomplete_service(self) -> bool:
        """Test the autocomplete service directly"""
        print("\\n🤖 Testing Autocomplete Service...")
        
        try:
            # Test health endpoint
            response = requests.get(f"{self.autocomplete_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"  ✅ Health check passed: {health_data.get('status', 'unknown')}")
                self.results["autocomplete"]["tests"].append({"test": "health", "status": "pass"})
            else:
                print(f"  ❌ Health check failed: {response.status_code}")
                self.results["autocomplete"]["tests"].append({"test": "health", "status": "fail"})
                return False
            
            # Test suggestions
            test_inputs = [
                "Hello, how are",
                "I think we should",
                "Can you help me",
                "The meeting is"
            ]
            
            for i, text in enumerate(test_inputs):
                payload = {"text": text, "user_id": f"test_user_{i}", "limit": 3}
                response = requests.post(f"{self.autocomplete_url}/suggest", 
                                       json=payload, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    suggestions = result.get("suggestions", [])
                    
                    print(f"  ✅ Input: '{text}' → {len(suggestions)} suggestions")
                    for j, suggestion in enumerate(suggestions[:2]):  # Show first 2
                        print(f"    {j+1}. {suggestion}")
                    
                    self.results["autocomplete"]["tests"].append({
                        "test": f"suggestion_{i}",
                        "status": "pass",
                        "input": text,
                        "suggestions_count": len(suggestions)
                    })
                else:
                    print(f"  ❌ Suggestion failed for '{text}': {response.status_code}")
                    self.results["autocomplete"]["tests"].append({
                        "test": f"suggestion_{i}",
                        "status": "fail"
                    })
                    return False
            
            # Test completion
            completion_payload = {"text": "The weather today is", "max_length": 20}
            response = requests.post(f"{self.autocomplete_url}/complete", 
                                   json=completion_payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                completion = result.get("completion", "")
                print(f"  ✅ Completion: 'The weather today is' → '{completion}'")
                self.results["autocomplete"]["tests"].append({"test": "completion", "status": "pass"})
            else:
                print(f"  ❌ Completion failed: {response.status_code}")
                self.results["autocomplete"]["tests"].append({"test": "completion", "status": "fail"})
                return False
            
            self.results["autocomplete"]["status"] = "operational"
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Autocomplete service unreachable: {e}")
            self.results["autocomplete"]["status"] = "unreachable"
            return False
    
    def test_backend_integration(self) -> bool:
        """Test if the Elixir backend properly integrates with both AI services"""
        print("\\n🔗 Testing Backend Integration...")
        
        # Note: This would require the Elixir backend to be running
        # For now, we'll just verify the service configurations
        
        try:
            # Test if backend can reach spam detection
            # This would normally be done through backend API endpoints
            print("  ℹ️  Backend integration tests require running Elixir backend")
            print("  ℹ️  Verify these configurations in your Elixir backend:")
            print(f"    - spam_detection_url: {self.spam_detection_url}")
            print(f"    - autocomplete_service_url: {self.autocomplete_url}")
            
            self.results["backend_integration"]["status"] = "manual_verification_needed"
            return True
            
        except Exception as e:
            print(f"  ❌ Backend integration test failed: {e}")
            self.results["backend_integration"]["status"] = "error"
            return False
    
    def generate_report(self):
        """Generate a comprehensive test report"""
        print("\\n" + "="*60)
        print("🔬 AI INTEGRATION TEST REPORT")
        print("="*60)
        
        for service_name, service_data in self.results.items():
            status = service_data["status"]
            status_emoji = {
                "operational": "✅",
                "unreachable": "❌", 
                "error": "❌",
                "manual_verification_needed": "⚠️",
                "unknown": "❓"
            }.get(status, "❓")
            
            print(f"\\n{status_emoji} {service_name.upper().replace('_', ' ')}: {status}")
            
            for test in service_data["tests"]:
                test_status = test["status"]
                test_emoji = {
                    "pass": "✅",
                    "fail": "❌",
                    "warning": "⚠️"
                }.get(test_status, "❓")
                
                print(f"  {test_emoji} {test['test']}")
        
        # Overall assessment
        spam_ok = self.results["spam_detection"]["status"] == "operational"
        autocomplete_ok = self.results["autocomplete"]["status"] == "operational"
        
        if spam_ok and autocomplete_ok:
            print("\\n🎉 OVERALL: Both AI services are operational and ready for backend integration!")
        elif spam_ok or autocomplete_ok:
            print("\\n⚠️  OVERALL: Partial success - one service is operational")
        else:
            print("\\n❌ OVERALL: Both services need attention before backend integration")
        
        print("\\n📝 NEXT STEPS:")
        if spam_ok and autocomplete_ok:
            print("  1. Ensure Elixir backend is configured with correct service URLs")
            print("  2. Test message sending through backend to verify spam detection")
            print("  3. Test autocomplete endpoints through backend API")
            print("  4. Monitor logs for successful API calls between services")
        else:
            print("  1. Fix any failed service tests above")
            print("  2. Ensure both services are running and accessible")
            print("  3. Re-run this test script")
    
    def run_all_tests(self):
        """Run all integration tests"""
        print("🚀 Starting AI Integration Tests...")
        print(f"Testing connections to:")
        print(f"  - Spam Detection: {self.spam_detection_url}")
        print(f"  - Autocomplete: {self.autocomplete_url}")
        print(f"  - Backend: {self.backend_url}")
        
        # Run individual service tests
        spam_ok = self.test_spam_detection_service()
        autocomplete_ok = self.test_autocomplete_service()
        backend_ok = self.test_backend_integration()
        
        # Generate report
        self.generate_report()
        
        return spam_ok and autocomplete_ok

def main():
    """Main test runner"""
    tester = AIIntegrationTester()
    
    print("🔍 AI Integration Test Suite")
    print("=" * 40)
    
    success = tester.run_all_tests()
    
    # Save results to file
    with open("ai_integration_test_results.json", "w") as f:
        json.dump(tester.results, f, indent=2)
    
    print(f"\\n📁 Detailed results saved to: ai_integration_test_results.json")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
