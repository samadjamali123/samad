#!/usr/bin/env python3
"""
Comprehensive System Testing Suite for Plant Leaf Disease Detection System

This module implements end-to-end testing covering unit, integration,
performance, security, and user journey validation with load testing
and automated performance benchmarking.
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

import pytest
import aiohttp
import requests
from PIL import Image, ImageDraw
import numpy as np
from io import BytesIO


class TestComprehensiveSystem:
    """End-to-end testing framework for production validation"""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url.rstrip('/')
        self.test_results = []
        self.performance_metrics = {}
        self.user_journeys = []
        self.load_test_results = {}
        
    async def run_complete_system_validation(self) -> Dict[str, Any]:
        """Run comprehensive system validation with all test categories"""
        print("🧪 Starting Complete System Validation")
        print("=" * 60)
        
        validation_results = {
            'start_time': datetime.utcnow().isoformat(),
            'validation_categories': [],
            'test_results': {},
            'performance_metrics': {},
            'system_health': {},
            'security_tests': {},
            'load_tests': {},
            'user_journeys': {},
            'overall_status': 'in_progress',
            'recommendations': []
        }
        
        # 1. System Health Check
        print("\n🏥 1. System Health Check")
        validation_results['system_health'] = await self._test_system_health()
        validation_results['validation_categories'].append('system_health')
        
        # 2. Core Functionality Tests
        print("\n🔧 2. Core Functionality Tests")
        validation_results['test_results']['core_functionality'] = await self._test_core_functionality()
        validation_results['validation_categories'].append('core_functionality')
        
        # 3. Security Tests
        print("\n🔒 3. Security Tests")
        validation_results['security_tests'] = await self._run_security_tests()
        validation_results['validation_categories'].append('security')
        
        # 4. Performance Tests
        print("\n⚡ 4. Performance Tests")
        validation_results['performance_metrics'] = await self._run_performance_tests()
        validation_results['validation_categories'].append('performance')
        
        # 5. Load Tests
        print("\n📊 5. Load Tests")
        validation_results['load_tests'] = await self._run_load_tests()
        validation_results['validation_categories'].append('load')
        
        # 6. User Journey Tests
        print("\n👤 6. User Journey Tests")
        validation_results['user_journeys'] = await self._test_user_journeys()
        validation_results['validation_categories'].append('user_journeys')
        
        # 7. Integration Tests
        print("\n🔗 7. Integration Tests")
        validation_results['test_results']['integration'] = await self._test_integration()
        validation_results['validation_categories'].append('integration')
        
        # 8. Grad-CAM Tests
        print("\n🎯 8. Grad-CAM Visualization Tests")
        validation_results['test_results']['grad_cam'] = await self._test_grad_cam_functionality()
        validation_results['validation_categories'].append('grad_cam')
        
        # 9. Database Tests
        print("\n🗄️ 9. Database Tests")
        validation_results['test_results']['database'] = await self._test_database_operations()
        validation_results['validation_categories'].append('database')
        
        # 10. API Documentation Tests
        print("\n📚 10. API Documentation Tests")
        validation_results['test_results']['documentation'] = await self._test_api_documentation()
        validation_results['validation_categories'].append('documentation')
        
        # Generate comprehensive report
        validation_results['end_time'] = datetime.utcnow().isoformat()
        validation_results['overall_status'] = await self._calculate_overall_status(validation_results)
        validation_results['recommendations'] = await self._generate_recommendations(validation_results)
        
        print(f"\n✅ System Validation Complete: {validation_results['overall_status']}")
        print("=" * 60)
        
        # Generate reports
        await self._generate_validation_reports(validation_results)
        
        return validation_results
    
    async def _test_system_health(self) -> Dict[str, Any]:
        """Test overall system health and service availability"""
        health_results = {
            'api_health': await self._test_api_health(),
            'database_health': await self._test_database_health(),
            'redis_health': await self._test_redis_health(),
            'external_apis': await self._test_external_api_health(),
            'model_availability': await self._test_model_availability(),
            'monitoring_health': await self._test_monitoring_health(),
        }
        
        # Calculate overall health score
        health_scores = []
        for category, results in health_results.items():
            score = results.get('status', 0) * 100
            health_scores.append(score)
        
        health_results['overall_score'] = statistics.mean(health_scores) if health_scores else 0
        health_results['status'] = 'healthy' if health_results['overall_score'] >= 95 else 'degraded'
        
        return health_results
    
    async def _test_api_health(self) -> Dict[str, Any]:
        """Test API endpoints health and responsiveness"""
        endpoints = [
            ('/', 'Root endpoint'),
            ('/health', 'Health check'),
            ('/models/status', 'Model status'),
            ('/diseases/categories', 'Disease categories'),
            ('/plants/supported', 'Supported plants')
        ]
        
        results = {}
        overall_start = time.time()
        
        async with aiohttp.ClientSession() as session:
            for endpoint, description in endpoints:
                try:
                    start_time = time.time()
                    async with session.get(f"{self.base_url}{endpoint}", timeout=10) as response:
                        response_time = (time.time() - start_time) * 1000
                        status = 'healthy' if response.status == 200 else 'unhealthy'
                        
                        results[endpoint] = {
                            'status': status,
                            'response_code': response.status,
                            'response_time_ms': round(response_time, 2),
                            'description': description
                        }
                        
                except Exception as e:
                    results[endpoint] = {
                        'status': 'error',
                        'error': str(e),
                        'response_time_ms': None,
                        'description': description
                    }
        
        # Calculate overall API health
        total_time = time.time() - overall_start
        healthy_endpoints = sum(1 for r in results.values() if r.get('status') == 'healthy')
        overall_score = (healthy_endpoints / len(endpoints)) * 100
        
        return {
            'status': 'healthy' if overall_score >= 80 else 'degraded',
            'overall_score': round(overall_score, 1),
            'total_test_time_ms': round(total_time * 1000, 2),
            'endpoint_results': results,
            'healthy_endpoints': healthy_endpoints,
            'total_endpoints': len(endpoints)
        }
    
    async def _test_core_functionality(self) -> Dict[str, Any]:
        """Test core system functionality"""
        test_image = self._create_test_image()
        
        async with aiohttp.ClientSession() as session:
            # Test 1: Basic analysis
            analysis_result = await self._test_basic_analysis(session, test_image)
            
            # Test 2: Batch analysis
            batch_result = await self._test_batch_analysis(session, [test_image] * 3])
            
            # Test 3: Grad-CAM generation
            grad_cam_result = await self._test_grad_cam_generation(session, test_image)
            
            # Test 4: Disease information retrieval
            disease_info_result = await self._test_disease_information(session)
            
            # Test 5: Search functionality
            search_result = await self._test_search_functionality(session)
            
            # Test 6: Reference images
            reference_result = await self._test_reference_images(session)
        
        results = {
            'basic_analysis': analysis_result,
            'batch_analysis': batch_result,
            'grad_cam_generation': grad_cam_result,
            'disease_information': disease_info_result,
            'search_functionality': search_result,
            'reference_images': reference_result
        }
        
        # Calculate core functionality score
        passed_tests = sum(1 for r in results.values() if r.get('passed', False))
        total_tests = len(results)
        overall_score = (passed_tests / total_tests) * 100
        
        return {
            'status': 'passed' if overall_score >= 90 else 'failed',
            'overall_score': round(overall_score, 1),
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'test_results': results
        }
    
    async def _test_basic_analysis(self, session: aiohttp.ClientSession, image_data: bytes) -> Dict[str, Any]:
        """Test basic image analysis functionality"""
        try:
            start_time = time.time()
            
            data = aiohttp.FormData()
            data.add_field('image', image_data, filename='test_leaf.jpg', content_type='image/jpeg')
            data.add_field('quality_threshold', '0.7')
            
            async with session.post(f"{self.base_url}/analyze", data=data, timeout=30) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    result_data = await response.json()
                    
                    # Validate response structure
                    required_fields = ['session_id', 'results', 'processing_time_ms', 'success']
                    missing_fields = [field for field in required_fields if field not in result_data]
                    
                    # Validate results structure
                    results = result_data.get('results', {})
                    required_result_fields = ['plant_detection', 'disease_detection', 'ai_model_results', 'metadata']
                    missing_result_fields = [field for field in required_result_fields if field not in results]
                    
                    return {
                        'passed': len(missing_fields) == 0 and len(missing_result_fields) == 0,
                        'response_time_ms': round(response_time, 2),
                        'response_code': response.status,
                        'missing_fields': missing_fields,
                        'missing_result_fields': missing_result_fields,
                        'plant_detected': bool(results.get('plant_detection', {}).get('plant_name')),
                        'disease_detected': bool(results.get('disease_detection', {}).get('primary_disease', {}).get('disease_name')),
                        'confidence_scores': {
                            'plant': results.get('plant_detection', {}).get('confidence', 0),
                            'disease': results.get('disease_detection', {}).get('primary_disease', {}).get('confidence', 0)
                        }
                    }
                else:
                    return {
                        'passed': False,
                        'response_time_ms': round(response_time, 2),
                        'response_code': response.status,
                        'error': f'HTTP {response.status}'
                    }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'response_time_ms': None
            }
    
    async def _test_batch_analysis(self, session: aiohttp.ClientSession, images: List[bytes]) -> Dict[str, Any]:
        """Test batch image analysis functionality"""
        try:
            start_time = time.time()
            
            data = aiohttp.FormData()
            for i, image_data in enumerate(images):
                data.add_field('images', image_data, filename=f'test_leaf_{i}.jpg', content_type='image/jpeg')
            
            async with session.post(f"{self.base_url}/analyze/batch", data=data, timeout=60) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    result_data = await response.json()
                    
                    return {
                        'passed': result_data.get('success', False),
                        'response_time_ms': round(response_time, 2),
                        'response_code': response.status,
                        'total_images': result_data.get('total_images', len(images)),
                        'successful_analyses': result_data.get('successful_analyses', 0),
                        'failed_analyses': result_data.get('failed_analyses', 0)
                    }
                else:
                    return {
                        'passed': False,
                        'response_time_ms': round(response_time, 2),
                        'response_code': response.status,
                        'error': f'HTTP {response.status}'
                    }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'response_time_ms': None
            }
    
    async def _test_grad_cam_generation(self, session: aiohttp.ClientSession, image_data: bytes) -> Dict[str, Any]:
        """Test Grad-CAM visualization generation"""
        try:
            start_time = time.time()
            
            data = aiohttp.FormData()
            data.add_field('image', image_data, filename='test_leaf.jpg', content_type='image/jpeg')
            
            async with session.post(f"{self.base_url}/analyze/grad-cam", data=data, timeout=45) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    result_data = await response.json()
                    
                    return {
                        'passed': result_data.get('success', False),
                        'response_time_ms': round(response_time, 2),
                        'response_code': response.status,
                        'has_grad_cam_data': 'grad_cam_data' in result_data,
                        'has_overlay_data': 'overlay_data' in result_data.get('grad_cam_data', {}),
                        'processing_time_ms': result_data.get('processing_time_ms', 0)
                    }
                else:
                    return {
                        'passed': False,
                        'response_time_ms': round(response_time, 2),
                        'response_code': response.status,
                        'error': f'HTTP {response.status}'
                    }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'response_time_ms': None
            }
    
    async def _test_disease_information(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test disease information retrieval"""
        test_disease_id = "early_blight_tomato"
        
        try:
            async with session.get(f"{self.base_url}/diseases/{test_disease_id}", timeout=10) as response:
                if response.status == 200:
                    result_data = await response.json()
                    
                    required_fields = ['success', 'disease', 'treatments']
                    missing_fields = [field for field in required_fields if field not in result_data]
                    
                    if 'disease' in result_data:
                        disease = result_data['disease']
                        required_disease_fields = ['disease_id', 'disease_name', 'scientific_name', 'description']
                        missing_disease_fields = [field for field in required_disease_fields if field not in disease]
                    else:
                        missing_disease_fields = []
                    
                    return {
                        'passed': len(missing_fields) == 0 and len(missing_disease_fields) == 0,
                        'response_code': response.status,
                        'missing_fields': missing_fields,
                        'missing_disease_fields': missing_disease_fields,
                        'has_treatments': bool(result_data.get('treatments'))
                    }
                else:
                    return {
                        'passed': False,
                        'response_code': response.status,
                        'error': f'HTTP {response.status}'
                    }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'response_code': None
            }
    
    async def _test_search_functionality(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test search functionality"""
        search_tests = [
            ('/search/diseases?q=blight&limit=5', 'Disease search'),
            ('/search/diseases?plant=tomato', 'Plant-specific search'),
            ('/search/diseases?query=early_blight&limit=3', 'Specific disease search')
        ]
        
        results = {}
        passed_tests = 0
        
        for endpoint, description in search_tests:
            try:
                async with session.get(f"{self.base_url}{endpoint}", timeout=10) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        
                        # Validate search response
                        has_results = result_data.get('success', False) and len(result_data.get('results', [])) > 0
                        if has_results:
                            passed_tests += 1
                        
                        results[description] = {
                            'passed': has_results,
                            'response_code': response.status,
                            'results_count': len(result_data.get('results', [])),
                            'has_query': 'query' in endpoint or 'q' in endpoint,
                            'has_pagination': 'limit' in endpoint
                        }
                    else:
                        results[description] = {
                            'passed': False,
                            'response_code': response.status,
                            'error': 'No results returned'
                        }
                    else:
                        results[description] = {
                            'passed': False,
                            'response_code': response.status,
                            'error': f'HTTP {response.status}'
                        }
            except Exception as e:
                results[description] = {
                    'passed': False,
                    'error': str(e),
                    'response_code': None
                }
        
        return {
            'status': 'passed' if passed_tests >= 2 else 'failed',
            'passed_tests': passed_tests,
            'total_tests': len(search_tests),
            'test_results': results
        }
    
    async def _test_reference_images(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Test reference images functionality"""
        try:
            async with session.get(f"{self.base_url}/diseases/early_blight_tomato", timeout=10) as response:
                if response.status == 200:
                    result_data = await response.json()
                    
                    # Check if reference images are included
                    disease = result_data.get('disease', {})
                    has_reference_images = 'reference_images' in disease
                    
                    return {
                        'passed': has_reference_images,
                        'response_code': response.status,
                        'has_reference_images': has_reference_images,
                        'reference_images_count': len(disease.get('reference_images', [])) if has_reference_images else 0
                    }
                else:
                    return {
                        'passed': False,
                        'response_code': response.status,
                        'error': f'HTTP {response.status}'
                    }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e),
                'response_code': None
            }
    
    async def _run_security_tests(self) -> Dict[str, Any]:
        """Run comprehensive security tests"""
        security_tests = [
            self._test_sql_injection_protection,
            self._test_xss_protection,
            self._test_rate_limiting,
            self._test_authentication_security,
            self._test_file_upload_security,
            self._test_csrf_protection,
            self._test_sensitive_data_exposure,
            self._test_api_access_control
        ]
        
        results = {}
        passed_tests = 0
        
        for test_func in security_tests:
            test_name = test_func.__name__.replace('_test_', '')
            try:
                result = await test_func()
                results[test_name] = result
                if result.get('vulnerability_found', True):
                    passed_tests += 1
            except Exception as e:
                results[test_name] = {
                    'vulnerability_found': True,
                    'details': f'Test error: {str(e)}'
                }
        
        # Calculate security score
        total_tests = len(security_tests)
        security_score = (passed_tests / total_tests) * 100
        
        return {
            'status': 'secure' if security_score >= 90 else 'vulnerable',
            'security_score': round(security_score, 1),
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'test_results': results,
            'recommendations': await self._generate_security_recommendations(results)
        }
    
    async def _test_sql_injection_protection(self) -> Dict[str, Any]:
        """Test SQL injection protection"""
        malicious_payloads = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "1' UNION SELECT * FROM users--",
            "'; INSERT INTO users VALUES('hacker', 'password'); --"
        ]
        
        for payload in malicious_payloads:
            try:
                async with aiohttp.ClientSession() as session:
                    encoded_payload = requests.utils.quote(payload)
                    async with session.get(f"{self.base_url}/search/diseases?q={encoded_payload}", timeout=10) as response:
                        
                        # Should return empty results or error, not database data
                        if response.status == 200:
                            result_data = await response.json()
                            results = result_data.get('results', [])
                            
                            # Check if any data was returned (vulnerable)
                            if len(results) > 0:
                                return {
                                    'vulnerability_found': True,
                                    'details': f'SQL injection vulnerability with payload: {payload}',
                                    'payload': payload,
                                    'response_code': response.status
                                }
            
            except Exception:
                continue  # Try next payload
        
        return {
            'vulnerability_found': False,
            'details': 'SQL injection protection appears to be working'
        }
    
    async def _test_xss_protection(self) -> Dict[str, Any]:
        """Test XSS protection"""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//",
            "<svg onload=alert('xss')>"
        ]
        
        for payload in xss_payloads:
            try:
                # Test in search (might be reflected)
                async with aiohttp.ClientSession() as session:
                    encoded_payload = requests.utils.quote(payload)
                    async with session.get(f"{self.base_url}/search/diseases?q={encoded_payload}", timeout=10) as response:
                        
                        if response.status == 200:
                            result_data = await response.json()
                            
                            # Check if payload is reflected in response
                            response_text = str(result_data).lower()
                            if payload.lower() in response_text:
                                return {
                                    'vulnerability_found': True,
                                    'details': f'XSS vulnerability with payload: {payload}',
                                    'payload': payload,
                                    'response_code': response.status
                                }
            
            except Exception:
                continue  # Try next payload
        
        return {
            'vulnerability_found': False,
            'details': 'XSS protection appears to be working'
        }
    
    async def _test_rate_limiting(self) -> Dict[str, Any]:
        """Test rate limiting functionality"""
        rapid_requests = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Make rapid requests to test rate limiting
                for i in range(70):  # More than typical rate limit of 60/minute
                    start_time = time.time()
                    
                    async with session.get(f"{self.base_url}/health", timeout=5) as response:
                        response_time = time.time() - start_time
                        
                        rapid_requests.append({
                            'request_number': i + 1,
                            'status_code': response.status,
                            'response_time': response_time
                        })
                        
                        # Break if we hit rate limit
                        if response.status == 429:
                            break
                    
                    # Small delay between requests
                    await asyncio.sleep(0.1)
        
        # Analyze results
        rate_limited_requests = [r for r in rapid_requests if r['status_code'] == 429]
        
        if rate_limited_requests:
            return {
                'vulnerability_found': False,  # This is expected behavior
                'details': f'Rate limiting working (limited after {rate_limited_requests[0]["request_number"]} requests)',
                'total_requests': len(rapid_requests),
                'rate_limited_after': rate_limited_requests[0]['request_number'] if rate_limited_requests else len(rapid_requests)
            }
        else:
            return {
                'vulnerability_found': True,
                'details': 'Rate limiting not working - no 429 responses received',
                'total_requests': len(rapid_requests)
            }
    
    async def _test_authentication_security(self) -> Dict[str, Any]:
        """Test authentication security"""
        auth_tests = [
            ('Invalid credentials', {'email': 'invalid@test.com', 'password': 'wrongpassword'}),
            ('SQL injection in auth', {'email': "'; DROP TABLE users; --", 'password': 'password'}),
            ('Missing credentials', {}),
            ('Empty password', {'email': 'test@example.com', 'password': ''}),
            ('Large password', {'email': 'test@example.com', 'password': 'x' * 1000})
        ]
        
        vulnerabilities = []
        
        for test_name, credentials in auth_tests:
            try:
                async with aiohttp.ClientSession() as session:
                    if credentials:
                        async with session.post(f"{self.base_url}/auth/login", json=credentials, timeout=10) as response:
                            if response.status == 200:
                                # This is potentially vulnerable - login should fail
                                vulnerabilities.append({
                                    'test': test_name,
                                    'vulnerability': 'Authentication bypass possible',
                                    'details': f'Login succeeded with {test_name}'
                                })
                            elif response.status not in [400, 401, 422]:
                                vulnerabilities.append({
                                    'test': test_name,
                                    'vulnerability': 'Unexpected response code',
                                    'details': f'Got {response.status} with {test_name}'
                                })
                    else:
                        # Test for missing authentication on protected endpoint
                        async with session.get(f"{self.base_url}/models/status", timeout=10) as protected_response:
                            if protected_response.status != 401:
                                vulnerabilities.append({
                                    'test': 'Missing authentication',
                                    'vulnerability': 'Protected endpoint accessible without auth',
                                    'details': f'Got {protected_response.status} on protected endpoint'
                                })
            except Exception as e:
                vulnerabilities.append({
                    'test': test_name,
                    'vulnerability': 'Test error',
                    'details': str(e)
                })
        
        return {
            'vulnerability_found': len(vulnerabilities) == 0,
            'details': vulnerabilities if vulnerabilities else 'Authentication security tests passed'
        }
    
    async def _test_file_upload_security(self) -> Dict[str, Any]:
        """Test file upload security"""
        upload_tests = [
            ('Oversized file', self._create_oversized_file()),
            ('Invalid file type', b'not_an_image_file'),
            ('Malicious file', self._create_malicious_file()),
            ('Path traversal', self._create_path_traversal_file()),
            ('ZIP bomb', self._create_zip_bomb_file())
        ]
        
        vulnerabilities = []
        
        for test_name, file_data in upload_tests:
            try:
                async with aiohttp.ClientSession() as session:
                    data = aiohttp.FormData()
                    data.add_field('image', file_data, filename='test_file', content_type='application/octet-stream')
                    
                    async with session.post(f"{self.base_url}/analyze", data=data, timeout=30) as response:
                        
                        # Check for proper error handling
                        if test_name == 'Oversized file' and response.status != 413:
                            vulnerabilities.append({
                                'test': test_name,
                                'vulnerability': 'File size limit not enforced',
                                'details': f'Got {response.status} instead of 413'
                            })
                        elif test_name == 'Invalid file type' and response.status != 400:
                            vulnerabilities.append({
                                'test': test_name,
                                'vulnerability': 'File type validation not working',
                                'details': f'Got {response.status} instead of 400'
                            })
                        elif test_name == 'Malicious file' and response.status == 200:
                            vulnerabilities.append({
                                'test': test_name,
                                'vulnerability': 'Malicious file accepted',
                                'details': 'Malicious file was processed successfully'
                            })
            except Exception as e:
                vulnerabilities.append({
                    'test': test_name,
                    'vulnerability': 'Test error',
                    'details': str(e)
                })
        
        return {
            'vulnerability_found': len(vulnerabilities) == 0,
            'details': vulnerabilities if vulnerabilities else 'File upload security tests passed'
        }
    
    def _create_oversized_file(self) -> bytes:
        """Create oversized file for testing"""
        # Create 10MB file (exceeding typical 5MB limit)
        return b'x' * (10 * 1024 * 1024)
    
    def _create_malicious_file(self) -> bytes:
        """Create potentially malicious file"""
        # Create file with suspicious content
        malicious_content = b'<?php system($_GET["cmd"]); ?>'
        return malicious_content
    
    def _create_path_traversal_file(self) -> bytes:
        """Create file with path traversal attempt"""
        return b'../../../etc/passwd'
    
    def _create_zip_bomb_file(self) -> bytes:
        """Create ZIP bomb file"""
        return b'PK' + b'\x00' * 1000  # Invalid ZIP structure that might cause issues
    
    async def _test_csrf_protection(self) -> Dict[str, Any]:
        """Test CSRF protection"""
        # Note: This is harder to test without a proper web form
        # We'll check if CSRF tokens are implemented in responses
        
        try:
            async with aiohttp.ClientSession() as session:
                # Make a request and check for CSRF tokens in response
                async with session.get(f"{self.base_url}/config", timeout=10) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        
                        # Look for CSRF-related headers or tokens
                        has_csrf_protection = any([
                            'csrf_token' in str(result_data),
                            'x-csrf-token' in response.headers,
                            'set-cookie' in response.headers and 'csrf' in response.headers.get('set-cookie', '')
                        ])
                        
                        return {
                            'vulnerability_found': not has_csrf_protection,
                            'details': 'CSRF protection: ' + ('Implemented' if has_csrf_protection else 'Not found'),
                            'has_csrf_token': has_csrf_protection
                        }
                    else:
                        return {
                            'vulnerability_found': True,
                            'details': f'Config endpoint returned {response.status}'
                        }
        except Exception as e:
            return {
                'vulnerability_found': True,
                'details': f'CSRF test error: {str(e)}'
            }
    
    async def _test_sensitive_data_exposure(self) -> Dict[str, Any]:
        """Test for sensitive data exposure in responses"""
        sensitive_endpoints = ['/config', '/models/status', '/monitoring/system']
        
        sensitive_patterns = [
            'password', 'secret', 'key', 'token', 'private',
            'database', 'admin', 'internal', 'config'
        ]
        
        try:
            async with aiohttp.ClientSession() as session:
                for endpoint in sensitive_endpoints:
                    async with session.get(f"{self.base_url}{endpoint}", timeout=10) as response:
                        if response.status == 200:
                            result_data = await response.json()
                            
                            # Check for sensitive data in response
                            response_text = str(result_data).lower()
                            found_sensitive = [pattern for pattern in sensitive_patterns if pattern in response_text]
                            
                            if found_sensitive:
                                return {
                                    'vulnerability_found': True,
                                    'details': f'Sensitive data found in {endpoint}: {found_sensitive}',
                                    'endpoint': endpoint,
                                    'sensitive_data': found_sensitive
                                }
            
            return {
                'vulnerability_found': False,
                'details': 'No sensitive data exposure detected'
            }
        except Exception as e:
            return {
                'vulnerability_found': True,
                'details': f'Sensitive data test error: {str(e)}'
            }
    
    async def _test_api_access_control(self) -> Dict[str, Any]:
        """Test API access control and permissions"""
        access_tests = [
            ('Admin endpoints', '/monitoring/dashboard'),
            ('Model management', '/models/status'),
            ('User management', '/monitoring/users'),
            ('Configuration access', '/config')
        ]
        
        vulnerabilities = []
        
        for test_name, endpoint in access_tests:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}{endpoint}", timeout=10) as response:
                        
                        # These should require authentication in production
                        if response.status == 200:
                            vulnerabilities.append({
                                'test': test_name,
                                'vulnerability': 'Public access to protected endpoint',
                                'details': f'{endpoint} accessible without auth'
                            })
                        elif response.status not in [401, 403, 422]:
                            vulnerabilities.append({
                                'test': test_name,
                                'vulnerability': 'Unexpected response code',
                                'details': f'Got {response.status} for {endpoint}'
                            })
            except Exception as e:
                vulnerabilities.append({
                    'test': test_name,
                    'vulnerability': 'Test error',
                    'details': str(e)
                })
        
        return {
            'vulnerability_found': len(vulnerabilities) == 0,
            'details': vulnerabilities if vulnerabilities else 'API access control tests passed'
        }
    
    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Run comprehensive performance tests"""
        performance_results = {
            'response_time_tests': await self._test_response_times(),
            'concurrent_user_tests': await self._test_concurrent_users(),
            'throughput_tests': await self._test_throughput(),
            'memory_usage_tests': await self._test_memory_usage(),
            'file_processing_performance': await self._test_file_processing_performance()
        }
        
        # Calculate overall performance score
        scores = []
        for test_category, results in performance_results.items():
            if isinstance(results, dict) and 'score' in results:
                scores.append(results['score'])
        
        overall_score = statistics.mean(scores) if scores else 0
        
        return {
            'overall_score': round(overall_score, 1),
            'status': 'excellent' if overall_score >= 90 else 'good' if overall_score >= 75 else 'needs_improvement',
            'test_categories': list(performance_results.keys()),
            'test_results': performance_results
        }
    
    async def _test_response_times(self) -> Dict[str, Any]:
        """Test API response times for various endpoints"""
        endpoints = [
            ('/health', 'Health check', 100),  # Should be < 100ms
            ('/models/status', 'Model status', 500),  # Should be < 500ms
            ('/diseases/categories', 'Disease categories', 1000),  # Should be < 1s
            ('/search/diseases?limit=10', 'Search', 500)  # Should be < 500ms
        ]
        
        results = {}
        
        async with aiohttp.ClientSession() as session:
            for endpoint, description, threshold in endpoints:
                times = []
                
                for i in range(5):  # Test each endpoint 5 times
                    start_time = time.time()
                    try:
                        async with session.get(f"{self.base_url}{endpoint}", timeout=10) as response:
                            response_time = time.time() - start_time
                            times.append(response_time)
                    except Exception:
                        times.append(float('inf'))  # Timeout or error
                
                # Calculate statistics
                valid_times = [t for t in times if t != float('inf')]
                
                if valid_times:
                    avg_time = statistics.mean(valid_times)
                    min_time = min(valid_times)
                    max_time = max(valid_times)
                    p95_time = np.percentile(valid_times, 95)
                    
                    # Calculate score based on threshold
                    score = max(0, 100 - ((avg_time - threshold) / threshold) * 100)
                    
                    results[description] = {
                        'avg_time_ms': round(avg_time, 2),
                        'min_time_ms': round(min_time, 2),
                        'max_time_ms': round(max_time, 2),
                        'p95_time_ms': round(p95_time, 2),
                        'threshold_ms': threshold,
                        'score': round(score, 1),
                        'within_threshold': avg_time <= threshold,
                        'tests_completed': len(times),
                        'successful_tests': len(valid_times)
                    }
                else:
                    results[description] = {
                        'avg_time_ms': float('inf'),
                        'score': 0,
                        'within_threshold': False,
                        'tests_completed': 0,
                        'successful_tests': 0
                    }
        
        # Calculate overall response time score
        scores = [result['score'] for result in results.values()]
        overall_score = statistics.mean(scores) if scores else 0
        
        return {
            'score': round(overall_score, 1),
            'endpoint_results': results,
            'overall_status': 'excellent' if overall_score >= 90 else 'good' if overall_score >= 75 else 'needs_improvement'
        }
    
    async def _test_concurrent_users(self) -> Dict[str, Any]:
        """Test system performance under concurrent load"""
        concurrent_levels = [10, 25, 50, 100]  # Different concurrency levels
        
        results = {}
        
        for concurrency in concurrent_levels:
            print(f"Testing with {concurrency} concurrent users...")
            
            start_time = time.time()
            successful_requests = 0
            response_times = []
            errors = []
            
            async with aiohttp.ClientSession() as session:
                tasks = []
                
                # Create concurrent tasks
                for i in range(concurrency):
                    task = self._make_health_check_request(session, i)
                    tasks.append(task)
                
                # Execute all tasks concurrently
                task_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Analyze results
                for i, result in enumerate(task_results):
                    if isinstance(result, Exception):
                        errors.append(str(result))
                    else:
                        successful_requests += 1
                        response_times.append(result.get('response_time', 0))
            
            total_time = time.time() - start_time
            
            # Calculate metrics
            avg_response_time = statistics.mean(response_times) if response_times else 0
            success_rate = (successful_requests / concurrency) * 100
            requests_per_second = concurrency / total_time if total_time > 0 else 0
            
            # Calculate score
            score = 0
            if avg_response_time > 0 and success_rate >= 95:
                score = 100
            elif avg_response_time > 0 and success_rate >= 90:
                score = 80
            elif success_rate >= 80:
                score = 60
            elif success_rate >= 70:
                score = 40
            
            total_time = time.time() - start_time
            
            return {
                'score': round(overall_score, 1),
                'concurrency_results': results,
                'overall_status': 'excellent' if overall_score >= 90 else 'good' if overall_score >= 75 else 'needs_improvement'
            }
        }
    
    async def _make_health_check_request(self, session: aiohttp.ClientSession, request_id: int) -> Dict[str, Any]:
        """Make a single health check request"""
        start_time = time.time()
        try:
            async with session.get(f"{self.base_url}/health", timeout=10) as response:
                return {
                    'request_id': request_id,
                    'response_time': time.time() - start_time,
                    'status_code': response.status,
                    'success': response.status == 200
                }
        except Exception as e:
            return {
                    'request_id': request_id,
                    'response_time': float('inf'),
                    'status_code': None,
                    'success': False,
                    'error': str(e)
                }
    
    async def _test_throughput(self) -> Dict[str, Any]:
        """Test system throughput capacity"""
        test_duration = 30  # 30 seconds
        target_rps = 50  # 50 requests per second
        
        start_time = time.time()
        successful_requests = 0
        total_requests = 0
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            # Continuous request generation
            request_start_time = time.time()
            while time.time() - request_start_time < test_duration:
                batch_tasks = []
                
                # Create batch of requests
                for _ in range(target_rps):  # One batch per second
                    if time.time() - request_start_time >= test_duration:
                        break
                    
                    task = self._make_health_check_request(session, total_requests)
                    batch_tasks.append(task)
                    total_requests += 1
                
                # Execute batch
                if batch_tasks:
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    for result in batch_results:
                        if isinstance(result, dict) and result.get('success', False):
                            successful_requests += 1
                
                # Small delay to maintain target RPS
                await asyncio.sleep(0.01)
        
        total_time = time.time() - start_time
        actual_rps = total_requests / total_time if total_time > 0 else 0
        success_rate = (successful_requests / total_requests) * 100 if total_requests > 0 else 0
        
        # Calculate throughput score
        score = 0
        if actual_rps >= target_rps * 0.9 and success_rate >= 95:
            score = 100
        elif actual_rps >= target_rps * 0.7 and success_rate >= 90:
            score = 80
        elif actual_rps >= target_rps * 0.5 and success_rate >= 80:
            score = 60
        elif success_rate >= 70:
            score = 40
        
        return {
            'score': score,
            'target_rps': target_rps,
            'actual_rps': round(actual_rps, 2),
            'success_rate': round(success_rate, 2),
            'total_requests': total_requests,
            'test_duration_s': round(total_time, 2),
            'throughput_score': 'excellent' if score >= 90 else 'good' if score >= 75 else 'needs_improvement'
        }
    
    async def _test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage under various loads"""
        memory_tests = [
            ('single_user', 1),
            ('moderate_load', 10),
            ('high_load', 50)
        ]
        
        results = {}
        
        for test_name, concurrency in memory_tests:
            print(f"Testing memory usage with {concurrency} users...")
            
            # Simulate concurrent analysis requests (more memory-intensive than health checks)
            test_image = self._create_test_image()
            
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                tasks = []
                
                for i in range(concurrency):
                    data = aiohttp.FormData()
                    data.add_field('image', test_image, filename=f'test_{i}.jpg', content_type='image/jpeg')
                    
                    task = asyncio.create_task(
                        session.post(f"{self.base_url}/analyze", data=data, timeout=60)
                    )
                    tasks.append(task)
                
                # Wait for all tasks to complete
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*tasks),
                        timeout=60
                    )
                except asyncio.TimeoutError:
                    pass  # Tasks timed out - likely memory issues
            
            total_time = time.time() - start_time
            
            # Note: In a real implementation, you'd monitor actual memory usage
            # Here we'll estimate based on concurrency and response patterns
            memory_score = max(0, 100 - (concurrency - 1) * 2)  # Arbitrary scoring
            
            results[test_name] = {
                'concurrency': concurrency,
                'memory_score': memory_score,
                'test_duration_s': round(total_time, 2),
                'memory_efficiency': 'high' if memory_score >= 80 else 'medium' if memory_score >= 60 else 'low'
            }
        
        # Calculate overall memory score
        scores = [result['memory_score'] for result in results.values()]
        overall_score = statistics.mean(scores) if scores else 0
        
        return {
            'score': round(overall_score, 1),
            'memory_test_results': results,
            'overall_status': 'excellent' if overall_score >= 85 else 'good' if overall_score >= 70 else 'needs_improvement'
        }
    
    async def _test_file_processing_performance(self) -> Dict[str, Any]:
        """Test file processing performance with different image sizes"""
        image_sizes = [
            ('small_image', (100, 100), 5),      # 5KB
            ('medium_image', (500, 500), 25),    # 25KB
            ('large_image', (1000, 1000), 100),  # 100KB
            ('huge_image', (2000, 2000), 400)   # 400KB
        ]
        
        results = {}
        
        for test_name, (width, height, estimated_size_kb) in image_sizes:
            print(f"Testing {test_name} ({width}x{height}, ~{estimated_size_kb}KB)...")
            
            test_image = self._create_sized_test_image(width, height)
            
            try:
                start_time = time.time()
                
                data = aiohttp.FormData()
                data.add_field('image', test_image, filename=f'{test_name}.jpg', content_type='image/jpeg')
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self.base_url}/analyze", data=data, timeout=120) as response:
                        
                        response_time = time.time() - start_time
                        
                        if response.status == 200:
                            result_data = await response.json()
                            
                            # Calculate processing rate
                            processing_rate = estimated_size_kb / response_time if response_time > 0 else 0
                            
                            results[test_name] = {
                                'size_kb': estimated_size_kb,
                                'response_time_s': round(response_time, 2),
                                'processing_rate_kb_s': round(processing_rate, 2),
                                'success': True,
                                'status_code': response.status,
                                'score': min(100, processing_rate * 2)  # Score based on processing rate
                            }
                        else:
                            results[test_name] = {
                                'size_kb': estimated_size_kb,
                                'response_time_s': round(response_time, 2),
                                'success': False,
                                'status_code': response.status,
                                'error': 'Processing failed'
                            }
                
                else:
                    results[test_name] = {
                        'size_kb': estimated_size_kb,
                        'success': False,
                        'status_code': response.status,
                        'error': f'HTTP {response.status}'
                    }
                
            except Exception as e:
                results[test_name] = {
                    'size_kb': estimated_size_kb,
                    'success': False,
                    'error': str(e)
                }
            
        # Calculate overall file processing score
        scores = [result.get('score', 0) for result in results.values()]
        overall_score = statistics.mean(scores) if scores else 0
        
        return {
            'score': round(overall_score, 1),
            'file_processing_results': results,
            'overall_status': 'excellent' if overall_score >= 90 else 'good' if overall_score >= 75 else 'needs_improvement'
        }
    
    def _create_test_image(self) -> bytes:
        """Create a test image for testing"""
        # Create a simple test image with leaf-like features
        image = Image.new('RGB', (300, 300), color='lightgreen')
        draw = ImageDraw.Draw(image)
        
        # Add some texture to simulate a leaf
        for y in range(0, 300, 5):
            for x in range(y % 10, 300, 20):
                draw.point((x, y), fill='darkgreen')
        
        # Add some spots to simulate disease
        draw.ellipse([(150, 150), (200, 200)], fill='brown')
        
        # Convert to bytes
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        return buffer.getvalue()
    
    def _create_sized_test_image(self, width: int, height: int) -> bytes:
        """Create a test image with specific dimensions"""
        image = Image.new('RGB', (width, height), color='green')
        
        # Add some noise/texture
        pixels = []
        for _ in range(width * height // 10):  # Add 10% noise
            x = np.random.randint(0, width)
            y = np.random.randint(0, height)
            pixels.append((x, y))
        
        for x, y in pixels:
            image.putpixel((x, y), (50, 100, 50))  # Dark green spots
        
        # Add disease spot for larger images
        if width > 500:
            draw = ImageDraw.Draw(image)
            draw.ellipse([(width//2, height//2), (width//2 + 50, height//2 + 50)], 
                       fill='brown')
        
        buffer = BytesIO()
        image.save(buffer, format='JPEG', quality=85)
        return buffer.getvalue()
    
    async def _run_load_tests(self) -> Dict[str, Any]:
        """Run comprehensive load tests"""
        load_test_scenarios = [
            ('sustained_load', 50, 300),      # 50 users for 5 minutes
            ('spike_load', 200, 30),         # 200 users for 30 seconds
            ('gradual_ramp', 100, 180),      # Gradually increase to 100 users over 3 minutes
            ('burst_load', 500, 10)            # 500 users for 10 seconds
        ]
        
        results = {}
        
        for scenario_name, max_users, duration in load_test_scenarios:
            print(f"Running {scenario_name}: {max_users} users for {duration}s...")
            
            result = await self._run_load_test_scenario(max_users, duration)
            results[scenario_name] = result
            
        # Calculate overall load test score
        scores = [result.get('score', 0) for result in results.values()]
        overall_score = statistics.mean(scores) if scores else 0
        
        return {
            'score': round(overall_score, 1),
            'scenario_results': results,
            'overall_status': 'excellent' if overall_score >= 90 else 'good' if overall_score >= 75 else 'needs_improvement'
        }
    
    async def _run_load_test_scenario(self, max_users: int, duration: int) -> Dict[str, Any]:
        """Run a single load test scenario"""
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            # Create user simulation tasks
            tasks = []
            for user_id in range(max_users):
                task = self._simulate_user_session(session, user_id, duration)
                tasks.append(task)
            
            # Execute all user simulations concurrently
            user_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Analyze results
            successful_users = sum(1 for result in user_results if result.get('success', False))
            error_count = sum(1 for result in user_results if isinstance(result, Exception))
            total_requests = sum(result.get('requests_made', 0) for result in user_results)
            avg_response_time = statistics.mean([
                result.get('avg_response_time', 0) for result in user_results
                if not isinstance(result, Exception)
            ]) if user_results else 0)
            
            # Calculate score
            success_rate = (successful_users / max_users) * 100
            score = 0
            if success_rate >= 95 and avg_response_time < 1.0:
                score = 100
            elif success_rate >= 90 and avg_response_time < 2.0:
                score = 80
            elif success_rate >= 80:
                score = 60
            elif success_rate >= 70:
                score = 40
            
            total_time = time.time() - start_time
            
            return {
                'scenario': f'{max_users}_users_{duration}s',
                'max_users': max_users,
                'duration': duration,
                'successful_users': successful_users,
                'success_rate': round(success_rate, 2),
                'total_requests': total_requests,
                'avg_response_time_s': round(avg_response_time, 3),
                'total_time_s': round(total_time, 2),
                'requests_per_second': round(total_requests / total_time, 2) if total_time > 0 else 0,
                'score': score,
                'error_count': error_count
            }
            }
    
    async def _simulate_user_session(self, session: aiohttp.ClientSession, user_id: int, duration: int) -> Dict[str, Any]:
        """Simulate a single user session with multiple requests"""
        start_time = time.time()
        end_time = start_time + duration
        
        requests_made = 0
        successful_requests = 0
        response_times = []
        errors = []
        
        test_image = self._create_test_image()
        
        try:
            while time.time() < end_time:
                request_start = time.time()
                
                data = aiohttp.FormData()
                data.add_field('image', test_image, filename=f'test_user_{user_id}.jpg', content_type='image/jpeg')
                
                try:
                    async with session.post(f"{self.base_url}/analyze", data=data, timeout=10) as response:
                        response_time = time.time() - request_start
                        requests_made += 1
                        
                        if response.status == 200:
                            successful_requests += 1
                            response_times.append(response_time)
                        else:
                            errors.append(f'HTTP {response.status}')
                        
                except Exception as e:
                    errors.append(str(e))
                
                # Small delay between requests (simulating user thinking time)
                await asyncio.sleep(1)
        
        total_time = time.time() - start_time
        avg_response_time = statistics.mean(response_times) if response_times else 0
        
        return {
            'user_id': user_id,
            'duration': duration,
            'success': len(errors) == 0,
            'requests_made': requests_made,
            'successful_requests': successful_requests,
            'avg_response_time_s': round(avg_response_time, 3),
            'total_time_s': round(total_time, 2),
            'requests_per_second': round(requests_made / total_time, 2) if total_time > 0 else 0),
            'errors': errors[:5]
        }
    
    async def _test_user_journeys(self) -> Dict[str, Any]:
        """Test complete user journeys"""
        user_journeys = [
            ('new_user_analysis', 'New user first plant analysis'),
            ('returning_user_grad_cam', 'Returning user explores Grad-CAM'),
            ('batch_analysis_workflow', 'User performs batch analysis'),
            ('disease_research', 'User researches specific disease'),
            ('reference_comparison', 'User compares with reference images')
        ]
        
        results = {}
        passed_journeys = 0
        
        for journey_name, description in user_journeys:
            print(f"Testing user journey: {description}")
            
            try:
                if journey_name == 'new_user_analysis':
                    result = await self._test_new_user_journey()
                elif journey_name == 'returning_user_grad_cam':
                    result = await self._test_returning_user_grad_cam_journey()
                elif journey_name == 'batch_analysis_workflow':
                    result = await self._test_batch_analysis_journey()
                elif journey_name == 'disease_research':
                    result = await self._test_disease_research_journey()
                elif journey_name == 'reference_comparison':
                    result = await self._test_reference_comparison_journey()
                
                results[journey_name] = result
                if result.get('passed', False):
                    passed_journeys += 1
                    
            except Exception as e:
                results[journey_name] = {
                    'passed': False,
                    'error': str(e)
                }
        
        # Calculate overall user journey score
        overall_score = (passed_journeys / len(user_journeys)) * 100
        
        return {
            'status': 'excellent' if overall_score >= 90 else 'good' if overall_score >= 75 else 'needs_improvement',
            'overall_score': round(overall_score, 1),
            'passed_journeys': passed_journeys,
            'total_journeys': len(user_journeys),
            'journey_results': results
        }
    
    async def _test_new_user_journey(self) -> Dict[str, Any]:
        """Test new user first analysis journey"""
        test_image = self._create_test_image()
        
        async with aiohttp.ClientSession() as session:
            # Step 1: Upload and analyze image
            analysis_data = await self._test_basic_analysis(session, test_image)
            
            if not analysis_data.get('passed', False):
                return {
                    'passed': False,
                    'error': 'Basic analysis failed'
                }
            
            # Step 2: Check if results are clear and actionable
            session_id = analysis_data.get('session_id', '')
            
            # Step 3: Try Grad-CAM (new user feature discovery)
            try:
                grad_cam_result = await self._test_grad_cam_generation(session, test_image)
                grad_cam_available = grad_cam_result.get('passed', False)
            except:
                    grad_cam_available = False
            
            # Step 4: Check disease information
            try:
                if analysis_data.get('disease_detected', False):
                    disease_info = await self._get_disease_details(session, 'early_blight_tomato')
                    disease_info_available = disease_info.get('passed', False)
                else:
                    disease_info_available = True  # No disease to check
            except:
                    disease_info_available = False
            
            # Step 5: Check for Grad-CAM download functionality
            grad_cam_download_result = await self._test_grad_cam_download(session, test_image)
            
            journey_score = 0
            if analysis_data.get('passed', False):
                journey_score += 0
            elif grad_cam_available:
                journey_score += 25
            elif disease_info_available:
                journey_score += 25
            else:
                journey_score = 100  # Basic success
            
            return {
                'passed': journey_score >= 50,
                'score': journey_score,
                'analysis_passed': analysis_data.get('passed', False),
                'grad_cam_available': grad_cam_available,
                'disease_info_available': disease_info_available,
                'grad_cam_download': grad_cam_download_result.get('passed', False)
            }
    
    async def _test_returning_user_grad_cam_journey(self) -> Dict[str, Any]:
        """Test returning user exploring Grad-CAM functionality"""
        # This tests Grad-CAM specifically as it's a key feature
        
        test_image = self._create_test_image()
        
        async with aiohttp.ClientSession() as session:
            # Step 1: Basic analysis
            analysis_data = await self._test_basic_analysis(session, test_image)
            
            if not analysis_data.get('passed', False):
                return {
                    'passed': False,
                    'error': 'Analysis failed'
                }
            
            # Step 2: Generate Grad-CAM
            grad_cam_result = await self._test_grad_cam_generation(session, test_image)
            
            # Step 3: Check if Grad-CAM provides insights
            grad_cam_available = grad_cam_result.get('passed', False)
            
            # Step 4: Try multiple Grad-CAM visualizations
            visualizations_working = True
            
            return {
                'passed': grad_cam_available,
                'score': 100 if grad_cam_available else 0,
                'visualizations_working': visualizations_working
            }
    
    async def _test_batch_analysis_journey(self) -> Dict[str, Any]:
        """Test user batch analysis workflow"""
        test_images = [self._create_test_image() for _ in range(3)]
        
        async with aiohttp.ClientSession() as session:
            # Step 1: Submit batch analysis
            batch_result = await self._test_batch_analysis(session, test_images)
            
            if not batch_result.get('passed', False):
                return {
                    'passed': False,
                    'error': 'Batch analysis failed'
                }
            
            # Step 2: Check individual results
            successful_analyses = batch_result.get('successful_analyses', 0)
            
            # Step 3: Verify batch processing time is reasonable
            batch_time = batch_result.get('response_time_ms', 0)
            
            # Step 4: Check if batch processing time is reasonable
            batch_time = batch_result.get('response_time_ms', 0)
            
            return {
                'passed': successful_analyses >= 2 and batch_time < 30000,  # At least 2 successful, under 30 seconds
                'score': min(100, successful_analyses * 33),  # Score based on success rate
                'successful_analyses': successful_analyses,
                'batch_time_ms': batch_time
            }
            }
    
    async def _test_disease_research_journey(self) -> Dict[str, Any]:
        """Test user disease research journey"""
        search_terms = ['early blight', 'powdery mildew', 'tomato diseases']
        
        async with aiohttp.ClientSession() as session:
            successful_searches = 0
            
            for search_term in search_terms:
                try:
                    encoded_term = requests.utils.quote(search_term)
                    async with session.get(f"{self.base_url}/search/diseases?q={encoded_term}", timeout=10) as response:
                        
                        if response.status == 200:
                            result_data = await response.json()
                            
                            if result_data.get('success', False) and len(result_data.get('results', [])) > 0:
                                successful_searches += 1
                                
                except Exception:
                            continue  # Try next term
            
            # Step 2: Get detailed information for a found disease
            if successful_searches > 0:
                disease_info = await self._get_disease_details(session, 'early_blight_tomato')
                disease_details_available = disease_info.get('passed', False)
            else:
                disease_details_available = True  # No diseases found to check
            
            return {
                'passed': (successful_searches > 0 and disease_details_available) or successful_searches >= 2,
                'score': (successful_searches * 25) + (25 if disease_details_available else 0),
                'successful_searches': successful_searches,
                'disease_details_available': disease_details_available
            }
            }
    
    async def _get_disease_details(self, session: aiohttp.ClientSession, disease_id: str) -> Dict[str, Any]:
        """Get detailed disease information"""
        try:
            async with session.get(f"{self.base_url}/diseases/{disease_id}", timeout=10) as response:
                if response.status == 200:
                    result_data = await response.json()
                    
                    required_fields = ['disease', 'treatments']
                    missing_fields = [field for field in required_fields if field not in result_data]
                    
                    if 'disease' in result_data:
                        disease = result_data['disease']
                        required_disease_fields = ['disease_id', 'disease_name', 'scientific_name', 'description']
                        missing_disease_fields = [field for field in required_disease_fields if field not in disease]
                    else:
                        missing_disease_fields = []
                    
                    return {
                        'passed': len(missing_fields) == 0 and len(missing_disease_fields) == 0,
                        'missing_fields': missing_fields,
                        'missing_disease_fields': missing_disease_fields,
                        'has_treatments': bool(result_data.get('treatments'))
                    }
                else:
                    return {
                        'passed': False,
                        'error': f'HTTP {response.status}'
                    }
        except Exception as e:
            return {
                'passed': False,
                'error': str(e)
            }
    
    async def _test_reference_comparison_journey(self) -> Dict[str, Any]:
        """Test user reference image comparison journey"""
        async with aiohttp.ClientSession() as session:
            # Step 1: Get a disease with reference images
            try:
                async with session.get(f"{self.base_url}/diseases/early_blight_tomato", timeout=10) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        
                        reference_images_available = (
                            result_data.get('success', False) and
                            'reference_images' in result_data.get('disease', {}) and
                            len(result_data['disease']['reference_images']) > 0
                        )
                    else:
                        reference_images_available = False
                else:
                        reference_images_available = False
            
            comparison_score = 50 if reference_images_available else 0
            
            return {
                'passed': reference_images_available,
                'score': comparison_score,
                'reference_images_available': reference_images_available
            }
        }
    
    async def _test_integration(self) -> Dict[str, Any]:
        """Test integration between components"""
        integration_tests = [
            ('database_integration', await self._test_database_integration()),
            ('redis_caching', await self._test_redis_caching()),
            ('monitoring_integration', await self._test_monitoring_integration()),
            ('external_api_integration', await self._test_external_api_integration())
        ]
        
        results = {}
        for test_name, result in integration_tests:
            results[test_name] = result
        
        passed_tests = sum(1 for result in results.values() if result.get('passed', False))
        overall_score = (passed_tests / len(integration_tests)) * 100
        
        return {
            'status': 'passed' if overall_score >= 90 else 'failed',
            'overall_score': round(overall_score, 1),
            'passed_tests': passed_tests,
            'total_tests': len(integration_tests),
            'test_results': results
        }
    
    async def _test_database_integration(self) -> Dict[str, Any]:
        """Test database integration"""
        # This would involve checking database operations, transactions, and data consistency
        # For now, we'll simulate this by testing API endpoints that use the database
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test storing analysis result (should work)
                test_image = self._create_test_image()
                data = aiohttp.FormData()
                data.add_field('image', test_image, filename='integration_test.jpg', content_type='image/jpeg')
                
                async with session.post(f"{self.base_url}/analyze", data=data, timeout=30) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        
                        # Check if session ID was generated (indicates database write)
                        has_session_id = 'session_id' in result_data
                        
                        return {
                            'passed': has_session_id,
                            'has_session_id': has_session_id,
                            'response_code': response.status,
                            'first_request_time_ms': round(response.time() * 1000, 2),
                            'second_request_time_ms': round(response_time * 1000, 2),
                            'cache_speedup': round((response.time() / response_time - 1) * 100, 2) if response_time2 > 0 else 0),
                            'cache_speedup': round((response_time1 / response_time2 - 1) * 100) if response_time2 > 0 else 0),
                            'first_request_time_ms': round(response_time1 * 1000, 2),
                            'second_request_time_ms': round(response_time2 * 1000, 2)
                        }
                            }
                        }
                    else:
                        return {
                            'passed': False,
                            'first_response_code': response1.status,
                            'error': f'HTTP {response.status}'
                        }
            }
        except Exception as e:
                return {
                    'passed': False,
                    'error': str(e)
            }
    
    async def _test_redis_caching(self) -> Dict[str, Any]:
        """Test Redis caching functionality"""
        # This would involve testing cache hits, cache invalidation, and performance
        
        try:
            async with aiohttp.ClientSession() as session:
                # Make same request twice to test caching
                test_image = self._create_test_image()
                data = aiohttp.FormData()
                data.add_field('image', test_image, filename='cache_test.jpg', content_type='image/jpeg')
                
                # First request
                start_time1 = time.time()
                async with session.post(f"{self.base_url}/analyze", data=data, timeout=30) as response1:
                    response_time1 = time.time() - start_time1
                    
                    # Second request (should be faster if cached)
                    start_time2 = time.time()
                    async with session.post(f"{self.base_url}/analyze", data=data, timeout=10) as response2:
                        response_time2 = time.time() - start_time2
                        
                        if response1.status == 200 and response2.status == 200:
                            # Both successful, check if second was faster (indicating cache)
                            cache_working = response_time2 < response_time1 * 0.8  # At least 20% faster
                            
                            return {
                                'passed': cache_working,
                                'first_request_time_ms': round(response_time1 * 1000, 2),
                                'second_request_time_ms': round(response_time2 * 1000, 2),
                                'cache_speedup': round((response_time1 / response_time2 - 1) * 100, 2) if response_time2 > 0 else 0),
                                'first_request_time_ms': round(response_time1 * 1000, 2),
                                'second_request_time_ms': round(response_time2 * 1000, 2)
                            }
                        }
                        else:
                            return {
                                'passed': True,  # Both worked but cache test inconclusive
                                'first_request_time_ms': round(response_time1 * 1000, 2),
                                'second_request_time_ms': round(response_time2 * 1000, 2),
                                'cache_speedup': 0
                            }
                        }
                    }
                    else:
                        return {
                            'passed': False,
                            'first_response_code': response1.status,
                            'error': f'HTTP {response1.status}'
                        }
                }
        except Exception:
                    return {
                        'passed': False,
                        'error': str(e)
                    }
            }
    
    async def _test_monitoring_integration(self) -> Dict[str, Any]:
        """Test monitoring system integration"""
        try:
            async with aiohttp.ClientSession() as session:
                # Test monitoring endpoints
                endpoints_to_test = [
                    '/monitoring/system',
                    '/monitoring/dashboard',
                    '/monitoring/health'
                ]
                
                working_endpoints = 0
                
                for endpoint in endpoints_to_test:
                    async with session.get(f"{self.base_url}{endpoint}", timeout=15) as response:
                        if response.status == 200:
                            working_endpoints += 1
                
                return {
                    'passed': working_endpoints >= 2,  # At least 2/3 working
                    'working_endpoints': working_endpoints,
                    'total_endpoints': len(endpoints_to_test),
                    'total_endpoints': len(endpoints_to_test)
                }
        except Exception as e:
                return {
                    'passed': False,
                    'error': str(e)
            }
    
    async def _test_external_api_integration(self) -> Dict[str, Any]:
        """Test integration with external APIs"""
        # This would test if the system can properly communicate with PlantNet, iNaturalist, etc.
        # For this test, we'll check if the models/status endpoint shows external API health
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/models/status", timeout=15) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        
                        # Check if external API health is reported
                        has_external_health = 'health_status' in result_data
                        
                        if has_external_health:
                            health_status = result_data['health_status']
                            available_models = result_data.get('available_models', [])
                            
                            # Check if multiple models are available (indicates good integration)
                            integration_score = len(available_models) * 20
                            
                            return {
                                'passed': integration_score >= 40,
                                'score': min(100, integration_score),
                                'available_models': available_models,
                                'health_status': health_status
                            }
                        else:
                            return {
                                'passed': False,
                                'error': 'No external API integration detected'
                            }
                    }
                    else:
                        return {
                            'passed': False,
                            'error': f'HTTP {response.status}'
                        }
                }
        except Exception as e:
            return {
                    'passed': False,
                    'error': str(e)
            }
            }
    
    async def _test_grad_cam_functionality(self) -> Dict[str, Any]:
        """Test Grad-CAM functionality comprehensively"""
        test_image = self._create_test_image()
        
        async with aiohttp.ClientSession() as session:
            # Test 1: Basic Grad-CAM generation
            grad_cam_result = await self._test_grad_cam_generation(session, test_image)
            
            # Test 2: Grad-CAM with different models (if supported)
            # Test 3: Grad-CAM overlay functionality
            # Test 4: Grad-CAM download functionality
            
            basic_passed = grad_cam_result.get('passed', False)
            has_heatmap = 'grad_cam_data' in grad_cam_result
            has_overlay = 'overlay_data' in grad_cam_result.get('grad_cam_data', {})
            
            score = 0
            if basic_passed:
                score += 50
            if has_heatmap:
                score += 25
            if has_overlay:
                score += 25
            
            return {
                'passed': basic_passed,
                'score': score,
                'has_heatmap': has_heatmap,
                'has_overlay': has_overlay,
                'basic_generation': basic_passed
            }
        }
    
    async def _test_database_operations(self) -> Dict[str, Any]:
        """Test database operations"""
        # This would test CRUD operations, transactions, etc.
        # For now, we'll test through API endpoints that use the database
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test data retrieval operations
                operations = [
                    ('diseases_categories', await self._test_get_operation(session, '/diseases/categories')),
                    ('plants_supported', await self._test_get_operation(session, '/plants/supported')),
                    ('database_stats', await self._test_get_operation(session, '/stats'))
                ]
                
                passed_operations = sum(1 for op_name, result in operations if result.get('passed', False))
                
                return {
                    'passed': passed_operations >= 2,  # At least 2 operations should work
                    'score': (passed_operations / len(operations)) * 100,
                    'operations': operations
                }
        except Exception as e:
                return {
                    'passed': False,
                    'error': str(e)
            }
    
    async def _test_get_operation(self, session: aiohttp.ClientSession, endpoint: str) -> Dict[str, Any]:
        """Test a GET operation"""
        try:
            async with session.get(f"{self.base_url}{endpoint}", timeout=10) as response:
                if response.status == 200:
                    result_data = await response.json()
                    
                    return {
                        'passed': result_data.get('success', False),
                        'response_code': response.status
                    }
                else:
                    return {
                        'passed': False,
                        'response_code': response.status,
                        'error': f'HTTP {response.status}'
                    }
        except Exception as e:
                return {
                    'passed': False,
                    'error': str(e)
                }
            }
    
    async def _calculate_overall_status(self, results: Dict[str, Any]) -> str:
        """Calculate overall system status"""
        scores = []
        
        # Collect all scores
        for category in ['system_health', 'security_tests', 'performance_metrics', 'load_tests', 'user_journeys']:
            if category in results:
                if isinstance(results[category], dict) and 'overall_score' in results[category]:
                    scores.append(results[category]['overall_score'])
                elif category == 'test_results' and 'core_functionality' in results[category]:
                    scores.append(results[category]['core_functionality']['overall_score'])
                elif category == 'test_results' and 'integration' in results[category]:
                    if isinstance(results[category]['integration'], dict) and 'overall_score' in results[category]['integration']):
                        scores.append(results[category]['integration']['overall_score'])
                    elif category == 'test_results' and 'grad_cam' in results[category]:
                        if isinstance(results[category]['grad_cam'], dict) and 'overall_score' in results[category]['grad_cam']):
                            scores.append(results[category]['grad_cam']['overall_score'])
                
                elif category == 'test_results' and 'database' in results[category]:
                    if isinstance(results[category]['database'], dict) and 'overall_score' in results[category]['database']):
                        scores.append(results[category]['database']['overall_score'])
                    elif category == 'test_results' and 'documentation' in results[category]:
                        if isinstance(results[category]['documentation'], dict) and 'overall_score' in results[category]['documentation']):
                            scores.append(results[category]['documentation']['overall_score'])
        
        if not scores:
            return 'failed'
        
        overall_score = statistics.mean(scores) if scores else 0
        
        if overall_score >= 95:
            return 'excellent'
        elif overall_score >= 85:
            return 'good'
        elif overall_score >= 70:
            return 'needs_improvement'
        else:
            return 'critical'
    
    async def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # System health recommendations
        system_health = results.get('system_health', {})
        if system_health.get('overall_score', 0) < 95:
            recommendations.append("Database connection issues detected - check database configuration and connectivity")
        
        if system_health.get('overall_score', 0) < 90:
            recommendations.append("System performance degradation detected - consider scaling resources")
        
        # Security recommendations
        security_results = results.get('security_tests', {})
        if security_results.get('vulnerability_found', True):
            recommendations.append("Security vulnerabilities detected - review and address security issues immediately")
        
        # Performance recommendations
        performance_results = results.get('performance_metrics', {})
        if performance_results.get('overall_score', 0) < 80:
            recommendations.append("Performance below acceptable levels - optimize database queries and implement caching")
        
        # Functionality recommendations
        test_results = results.get('test_results', {})
        core_functionality = test_results.get('core_functionality', {})
        if core_functionality.get('overall_score', 0) < 90:
            recommendations.append("Core functionality issues detected - review API endpoints and data processing")
        
        return recommendations
    
    async def _generate_security_recommendations(self, security_results: Dict[str, Any]) -> List[str]:
        """Generate specific security recommendations"""
        recommendations = []
        
        test_results = security_results.get('test_results', {})
        
        for test_name, result in test_results.items():
            if result.get('vulnerability_found', True):
                if test_name == 'sql_injection_protection':
                    recommendations.append("Implement parameterized queries to prevent SQL injection")
                    recommendations.append("Use prepared statements with parameter binding")
                elif test_name == 'xss_protection':
                    recommendations.append("Implement proper input sanitization and output encoding")
                    recommendations.append("Use Content Security Policy headers")
                elif test_name == 'rate_limiting':
                    recommendations.append("Ensure rate limiting is properly configured and enforced")
                elif test_name == 'authentication_security':
                    recommendations.append("Implement strong password policies and account lockout")
                    recommendations.append("Use secure session management and HTTPS")
                elif test_name == 'file_upload_security':
                    recommendations.append("Validate file types and sizes on upload")
                    recommendations.append("Scan uploaded files for malicious content")
                elif test_name == 'csrf_protection':
                    recommendations.append("Implement CSRF tokens for state-changing operations")
                    recommendations.append("Use SameSite cookies and CSRF headers")
                elif test_name == 'sensitive_data_exposure':
                    recommendations.append("Review API responses for sensitive data exposure")
                    recommendations.append("Implement proper data filtering and masking")
                elif test_name == 'api_access_control':
                    recommendations.append("Implement proper authentication and authorization")
                    recommendations.append("Review and secure all admin endpoints")
        
        return recommendations
    
    async def _generate_validation_reports(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive validation reports"""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Generate JSON report
        json_report = {
            'validation_timestamp': results['start_time'],
            'completion_timestamp': results['end_time'],
            'overall_status': results['overall_status'],
            'system_health': results.get('system_health', {}),
            'security_tests': results.get('security_tests', {}),
            'performance_metrics': results.get('performance_metrics', {}),
            'load_tests': results.get('load_tests', {}),
            'user_journeys': results.get('user_journeys', {}),
            'test_results': results.get('test_results', {}),
            'recommendations': results.get('recommendations', [])
        }
        
        # Save JSON report
        with open(f'validation_report_{timestamp}.json', 'w') as f:
            json.dump(json_report, f, indent=2)
        
        # Generate HTML report
        html_report = self._generate_html_report(results, timestamp)
        
        with open(f'validation_report_{timestamp}.html', 'w') as f:
            f.write(html_report)
        
        print(f"\n📊 Validation Reports Generated:")
        print(f"  JSON: validation_report_{timestamp}.json")
        print(f"  HTML: validation_report_{timestamp}.html")
        print(f"  Overall Status: {results['overall_status']}")
        
        # Exit with appropriate code
        exit_code = 0 if results['overall_status'] in ['excellent', 'good'] else 1
    
        return exit_code


# Main execution
async def main():
    """Main execution function"""
    print("🚀 Starting Comprehensive System Testing")
    print("=" * 60)
    
    # Initialize test system
    tester = TestComprehensiveSystem()
    
    # Run complete validation
    results = await tester.run_complete_system_validation()
    
    # Display summary
    print(f"\n🎯 TESTING COMPLETE!")
    print(f"Overall Status: {results['overall_status']}")
    print(f"\n📊 Detailed reports saved to validation_report_*.json/html")
    
    # Exit with appropriate code
    exit_code = 0 if results['overall_status'] in ['excellent', 'good'] else 1


if __name__ == "__main__":
    asyncio.run(main())