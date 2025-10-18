#!/usr/bin/env python3
"""
Comprehensive test runner for the AI-based infant mortality prediction system.

This script runs all unit tests, generates coverage reports, and provides
detailed test results. It ensures test isolation from production data.
"""

import os
import sys
import subprocess
import argparse
import json
import time
from datetime import datetime
import shutil
import tempfile

# Test configuration
TEST_CONFIG = {
    'test_directories': ['tests/'],
    'test_files': [
        'tests/test_api.py',
        'tests/test_models.py', 
        'tests/test_logger.py',
        'tests/test_data_ingestion.py'
    ],
    'coverage_threshold': 80,
    'output_formats': ['terminal', 'json', 'html'],
    'isolated_test_environment': True
}

class TestRunner:
    """
    Comprehensive test runner with environment isolation and detailed reporting
    """
    
    def __init__(self, config=None):
        self.config = config or TEST_CONFIG
        self.start_time = None
        self.results = {
            'summary': {},
            'test_results': {},
            'coverage': {},
            'errors': [],
            'warnings': []
        }
        self.temp_dir = None
        self.original_cwd = os.getcwd()
        
    def setup_test_environment(self):
        """Setup isolated test environment"""
        print("🔧 Setting up isolated test environment...")
        
        try:
            # Create temporary directory for test isolation
            if self.config.get('isolated_test_environment', True):
                self.temp_dir = tempfile.mkdtemp(prefix='test_env_')
                print(f"   Created isolated environment: {self.temp_dir}")
                
                # Create necessary directories
                test_dirs = ['models', 'logs', 'data', 'tests']
                for dir_name in test_dirs:
                    os.makedirs(os.path.join(self.temp_dir, dir_name), exist_ok=True)
                
                # Copy test files to isolated environment
                src_dirs = ['src', 'api', 'tests']
                for src_dir in src_dirs:
                    if os.path.exists(src_dir):
                        dest_dir = os.path.join(self.temp_dir, src_dir)
                        shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
                        print(f"   Copied {src_dir} to isolated environment")
                
                # Copy configuration files
                config_files = ['.streamlit/config.toml', 'app.py']
                for config_file in config_files:
                    if os.path.exists(config_file):
                        dest_path = os.path.join(self.temp_dir, config_file)
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        shutil.copy2(config_file, dest_path)
                        print(f"   Copied {config_file}")
                
                # Change to test environment
                os.chdir(self.temp_dir)
                print(f"   Changed working directory to: {self.temp_dir}")
            
            # Set environment variables for testing
            os.environ['TESTING'] = 'true'
            os.environ['LOG_LEVEL'] = 'DEBUG'
            
            # Ensure pytest is available
            try:
                import pytest
                print("   ✅ pytest is available")
            except ImportError:
                print("   ❌ pytest not found. Installing...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pytest', 'pytest-cov'])
                print("   ✅ pytest installed")
            
            print("✅ Test environment setup complete\n")
            return True
            
        except Exception as e:
            self.results['errors'].append(f"Environment setup failed: {str(e)}")
            print(f"❌ Environment setup failed: {str(e)}")
            return False
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        try:
            # Change back to original directory
            os.chdir(self.original_cwd)
            
            # Clean up temporary directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up test environment: {self.temp_dir}")
            
            # Clean up environment variables
            if 'TESTING' in os.environ:
                del os.environ['TESTING']
            if 'LOG_LEVEL' in os.environ:
                del os.environ['LOG_LEVEL']
            
        except Exception as e:
            print(f"⚠️ Cleanup warning: {str(e)}")
    
    def check_dependencies(self):
        """Check if all required dependencies are available"""
        print("🔍 Checking test dependencies...")
        
        required_packages = [
            'pytest', 'pandas', 'numpy', 'scikit-learn', 
            'matplotlib', 'seaborn', 'flask', 'requests', 'joblib'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                missing_packages.append(package)
                print(f"   ❌ {package}")
        
        if missing_packages:
            print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
            print("Installing missing packages...")
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install'
                ] + missing_packages)
                print("✅ All dependencies installed\n")
            except subprocess.CalledProcessError as e:
                self.results['errors'].append(f"Failed to install dependencies: {str(e)}")
                return False
        else:
            print("✅ All dependencies available\n")
        
        return True
    
    def run_unit_tests(self):
        """Run all unit tests"""
        print("🧪 Running unit tests...")
        
        test_results = {}
        
        for test_file in self.config['test_files']:
            if not os.path.exists(test_file):
                print(f"   ⚠️ Test file not found: {test_file}")
                self.results['warnings'].append(f"Test file not found: {test_file}")
                continue
            
            print(f"   Running {test_file}...")
            
            try:
                # Run pytest with verbose output and coverage
                cmd = [
                    sys.executable, '-m', 'pytest', 
                    test_file,
                    '-v',
                    '--tb=short',
                    '--cov=src',
                    '--cov=api',
                    '--cov-report=term-missing',
                    '--json-report',
                    f'--json-report-file=test_results_{os.path.basename(test_file)}.json'
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout per test file
                )
                
                test_results[test_file] = {
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'success': result.returncode == 0
                }
                
                if result.returncode == 0:
                    print(f"   ✅ {test_file} passed")
                else:
                    print(f"   ❌ {test_file} failed")
                    print(f"      Error: {result.stderr[:200]}...")
                
            except subprocess.TimeoutExpired:
                print(f"   ⏰ {test_file} timed out")
                test_results[test_file] = {
                    'return_code': -1,
                    'success': False,
                    'error': 'Test timed out'
                }
            
            except Exception as e:
                print(f"   ❌ {test_file} error: {str(e)}")
                test_results[test_file] = {
                    'return_code': -1,
                    'success': False,
                    'error': str(e)
                }
        
        self.results['test_results'] = test_results
        
        # Calculate summary
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results.values() if r.get('success', False))
        failed_tests = total_tests - passed_tests
        
        print(f"\n📊 Test Summary:")
        print(f"   Total test files: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {failed_tests}")
        print(f"   Success rate: {(passed_tests/total_tests*100):.1f}%\n")
        
        return passed_tests == total_tests
    
    def run_integration_tests(self):
        """Run integration tests"""
        print("🔗 Running integration tests...")
        
        integration_commands = [
            # Test API server startup (mock)
            [sys.executable, '-c', 'import api.app; print("API import successful")'],
            
            # Test model training pipeline (mock)
            [sys.executable, '-c', 'from src.models import ModelTrainer; print("Model import successful")'],
            
            # Test data ingestion (mock)
            [sys.executable, '-c', 'from src.data_ingestion import DataIngestion; print("Data ingestion import successful")'],
            
            # Test logging system
            [sys.executable, '-c', 'from src.logger import setup_logger; logger = setup_logger(); logger.info("Test"); print("Logging system working")']
        ]
        
        integration_results = {}
        
        for i, cmd in enumerate(integration_commands):
            test_name = f"integration_test_{i+1}"
            print(f"   Running {test_name}...")
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                integration_results[test_name] = {
                    'return_code': result.returncode,
                    'success': result.returncode == 0,
                    'output': result.stdout,
                    'error': result.stderr
                }
                
                if result.returncode == 0:
                    print(f"   ✅ {test_name} passed")
                else:
                    print(f"   ❌ {test_name} failed: {result.stderr}")
                
            except subprocess.TimeoutExpired:
                print(f"   ⏰ {test_name} timed out")
                integration_results[test_name] = {
                    'return_code': -1,
                    'success': False,
                    'error': 'Timeout'
                }
            except Exception as e:
                print(f"   ❌ {test_name} error: {str(e)}")
                integration_results[test_name] = {
                    'return_code': -1,
                    'success': False,
                    'error': str(e)
                }
        
        self.results['integration_results'] = integration_results
        
        passed_integration = sum(1 for r in integration_results.values() if r.get('success', False))
        total_integration = len(integration_results)
        
        print(f"\n📊 Integration Test Summary:")
        print(f"   Total: {total_integration}")
        print(f"   Passed: {passed_integration}")
        print(f"   Failed: {total_integration - passed_integration}\n")
        
        return passed_integration == total_integration
    
    def generate_coverage_report(self):
        """Generate test coverage report"""
        print("📈 Generating coverage report...")
        
        try:
            # Run coverage analysis
            cmd = [
                sys.executable, '-m', 'pytest',
                '--cov=src',
                '--cov=api', 
                '--cov-report=json',
                '--cov-report=html:htmlcov',
                '--quiet'
            ] + self.config['test_files']
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Read coverage JSON report
            if os.path.exists('coverage.json'):
                with open('coverage.json', 'r') as f:
                    coverage_data = json.load(f)
                    
                self.results['coverage'] = coverage_data
                
                # Extract summary
                total_coverage = coverage_data['totals']['percent_covered']
                print(f"   📊 Overall coverage: {total_coverage:.1f}%")
                
                if total_coverage >= self.config['coverage_threshold']:
                    print(f"   ✅ Coverage meets threshold ({self.config['coverage_threshold']}%)")
                else:
                    print(f"   ⚠️ Coverage below threshold ({self.config['coverage_threshold']}%)")
                    self.results['warnings'].append(f"Coverage {total_coverage:.1f}% below threshold {self.config['coverage_threshold']}%")
                
                # Show per-file coverage
                print("\n   📋 Coverage by file:")
                for filename, file_data in coverage_data['files'].items():
                    file_coverage = file_data['summary']['percent_covered']
                    print(f"      {filename}: {file_coverage:.1f}%")
                
            else:
                print("   ⚠️ Coverage report not generated")
                self.results['warnings'].append("Coverage report not found")
                
        except Exception as e:
            print(f"   ❌ Coverage report failed: {str(e)}")
            self.results['errors'].append(f"Coverage report failed: {str(e)}")
    
    def generate_reports(self):
        """Generate comprehensive test reports"""
        print("📝 Generating test reports...")
        
        # Generate JSON report
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'duration': time.time() - self.start_time if self.start_time else 0,
            'summary': self.calculate_summary(),
            'results': self.results,
            'environment': {
                'python_version': sys.version,
                'platform': sys.platform,
                'working_directory': os.getcwd()
            }
        }
        
        # Save JSON report
        with open('test_report.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        print("   ✅ JSON report saved: test_report.json")
        
        # Generate HTML report (simple)
        html_report = self.generate_html_report(report_data)
        with open('test_report.html', 'w') as f:
            f.write(html_report)
        print("   ✅ HTML report saved: test_report.html")
        
        # Print summary to terminal
        self.print_final_summary(report_data)
    
    def calculate_summary(self):
        """Calculate test run summary"""
        summary = {
            'total_test_files': len(self.results.get('test_results', {})),
            'passed_test_files': sum(1 for r in self.results.get('test_results', {}).values() if r.get('success')),
            'failed_test_files': 0,
            'total_integration_tests': len(self.results.get('integration_results', {})),
            'passed_integration_tests': sum(1 for r in self.results.get('integration_results', {}).values() if r.get('success')),
            'errors': len(self.results.get('errors', [])),
            'warnings': len(self.results.get('warnings', [])),
            'overall_success': True
        }
        
        summary['failed_test_files'] = summary['total_test_files'] - summary['passed_test_files']
        summary['overall_success'] = (
            summary['failed_test_files'] == 0 and 
            len(self.results.get('errors', [])) == 0
        )
        
        return summary
    
    def generate_html_report(self, report_data):
        """Generate HTML test report"""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Report - AI Infant Mortality Prediction System</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 8px; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px; }}
                .success {{ color: #28a745; }}
                .error {{ color: #dc3545; }}
                .warning {{ color: #ffc107; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .test-passed {{ background-color: #d4edda; }}
                .test-failed {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧪 Test Report</h1>
                <h2>AI Infant Mortality Prediction System</h2>
                <p><strong>Generated:</strong> {report_data['timestamp']}</p>
                <p><strong>Duration:</strong> {report_data['duration']:.2f} seconds</p>
            </div>
            
            <div class="summary">
                <h3>📊 Summary</h3>
                <p><strong>Test Files:</strong> {report_data['summary']['passed_test_files']}/{report_data['summary']['total_test_files']} passed</p>
                <p><strong>Integration Tests:</strong> {report_data['summary']['passed_integration_tests']}/{report_data['summary']['total_integration_tests']} passed</p>
                <p><strong>Errors:</strong> {report_data['summary']['errors']}</p>
                <p><strong>Warnings:</strong> {report_data['summary']['warnings']}</p>
                <p class="{'success' if report_data['summary']['overall_success'] else 'error'}">
                    <strong>Overall Result:</strong> {'✅ PASSED' if report_data['summary']['overall_success'] else '❌ FAILED'}
                </p>
            </div>
            
            <h3>📋 Detailed Results</h3>
            <table>
                <tr><th>Test File</th><th>Status</th><th>Details</th></tr>
        """
        
        for test_file, result in report_data['results'].get('test_results', {}).items():
            status_class = 'test-passed' if result.get('success') else 'test-failed'
            status_text = '✅ PASSED' if result.get('success') else '❌ FAILED'
            
            html += f"""
                <tr class="{status_class}">
                    <td>{test_file}</td>
                    <td>{status_text}</td>
                    <td>{result.get('error', 'Success') if not result.get('success') else 'All tests passed'}</td>
                </tr>
            """
        
        html += """
            </table>
        </body>
        </html>
        """
        
        return html
    
    def print_final_summary(self, report_data):
        """Print final test summary"""
        summary = report_data['summary']
        
        print("\n" + "="*60)
        print("🎯 FINAL TEST SUMMARY")
        print("="*60)
        print(f"📁 Test Files:        {summary['passed_test_files']}/{summary['total_test_files']} passed")
        print(f"🔗 Integration Tests: {summary['passed_integration_tests']}/{summary['total_integration_tests']} passed")
        print(f"⚠️  Warnings:         {summary['warnings']}")
        print(f"❌ Errors:           {summary['errors']}")
        print(f"⏱️  Duration:         {report_data['duration']:.2f} seconds")
        
        if 'coverage' in self.results and 'totals' in self.results['coverage']:
            coverage = self.results['coverage']['totals']['percent_covered']
            print(f"📊 Coverage:          {coverage:.1f}%")
        
        print("\n" + ("✅ ALL TESTS PASSED!" if summary['overall_success'] else "❌ SOME TESTS FAILED!"))
        print("="*60)
        
        if self.results.get('errors'):
            print("\n❌ ERRORS:")
            for error in self.results['errors']:
                print(f"   • {error}")
        
        if self.results.get('warnings'):
            print("\n⚠️ WARNINGS:")
            for warning in self.results['warnings']:
                print(f"   • {warning}")
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("🚀 Starting comprehensive test run...")
        print("="*60)
        
        self.start_time = time.time()
        
        try:
            # Setup environment
            if not self.setup_test_environment():
                return False
            
            # Check dependencies
            if not self.check_dependencies():
                return False
            
            # Run unit tests
            unit_success = self.run_unit_tests()
            
            # Run integration tests
            integration_success = self.run_integration_tests()
            
            # Generate coverage report
            self.generate_coverage_report()
            
            # Generate reports
            self.generate_reports()
            
            return unit_success and integration_success
            
        finally:
            # Always cleanup
            self.cleanup_test_environment()


def main():
    """Main entry point for test runner"""
    parser = argparse.ArgumentParser(
        description='Run comprehensive tests for AI infant mortality prediction system'
    )
    
    parser.add_argument(
        '--coverage-threshold',
        type=int,
        default=80,
        help='Minimum coverage threshold (default: 80%%)'
    )
    
    parser.add_argument(
        '--no-isolation',
        action='store_true',
        help='Disable test environment isolation'
    )
    
    parser.add_argument(
        '--specific-test',
        type=str,
        help='Run specific test file only'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick tests only (skip integration tests)'
    )
    
    args = parser.parse_args()
    
    # Update configuration based on arguments
    config = TEST_CONFIG.copy()
    config['coverage_threshold'] = args.coverage_threshold
    config['isolated_test_environment'] = not args.no_isolation
    
    if args.specific_test:
        config['test_files'] = [args.specific_test]
    
    # Initialize and run tests
    runner = TestRunner(config)
    
    try:
        success = runner.run_all_tests()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⚠️ Test run interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test run failed with unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
