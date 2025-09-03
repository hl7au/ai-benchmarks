#!/usr/bin/env python3
"""
FHIR AI Benchmark Framework
A tool for evaluating AI models on FHIR resource generation tasks using OpenRouter.
"""

import asyncio
import json
import yaml
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import aiohttp
import ssl
import certifi
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """Represents a single benchmark test case."""
    id: str
    name: str
    role: str
    system_prompt: str
    prompt: str
    expected_resource_type: Optional[str] = None
    category: str = "general"
    difficulty: str = "medium"
    scoring_criteria: Dict[str, float] = None
    metadata: Dict[str, Any] = None

@dataclass
class ModelConfig:
    """Configuration for an AI model to test via OpenRouter."""
    name: str
    model_id: str  # OpenRouter model ID (e.g., "anthropic/claude-3.5-sonnet")
    max_tokens: int = 4000
    temperature: float = 0.1
    top_p: float = 1.0

@dataclass
class TestResult:
    """Results from running a test case against a model."""
    test_id: str
    model_name: str
    response: str
    fhir_validation: Dict[str, Any]
    scores: Dict[str, float]
    total_score: float
    execution_time: float
    timestamp: datetime
    error: Optional[str] = None

class FHIRValidator:
    """Wrapper for FHIR validator MCP."""
    
    def __init__(self, mcp_server_path: str = "npx mcp-fhir-tools"):
        self.server_path = mcp_server_path.split()
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        server_params = StdioServerParameters(
            command=self.server_path[0],  # Pass as string, not list
            args=self.server_path[1:],
            env=None
        )
        
        # Correct pattern:
        # 1. Create stdio client context manager
        self.stdio_client = stdio_client(server_params)
        read_stream, write_stream = await self.stdio_client.__aenter__()

        # 2. Create session with the streams  
        self.session = ClientSession(read_stream, write_stream)
        await self.session.__aenter__()

        # 3. Initialize the connection
        await self.session.initialize()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
    
    async def validate_resource(self, fhir_json: str) -> Dict[str, Any]:
        """Validate FHIR resource and return validation results."""
        try:
            # Parse JSON to ensure it's valid
            resource = json.loads(fhir_json)
            
            # Call FHIR validator via MCP
            result = await self.session.call_tool("validate_resource", {
                "resource": resource
            })
            
            return {
                "valid": result.get("valid", False),
                "errors": result.get("errors", []),
                "warnings": result.get("warnings", []),
                "resource_type": resource.get("resourceType", "unknown")
            }
            
        except json.JSONDecodeError as e:
            return {
                "valid": False,
                "errors": [f"Invalid JSON: {str(e)}"],
                "warnings": [],
                "resource_type": "unknown"
            }
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
                "resource_type": "unknown"
            }

class OpenRouterClient:
    """Client for OpenRouter API."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.session = None
    
    async def __aenter__(self):
        """Create HTTP session with proper SSL context."""
        # Create proper SSL context for macOS certificate issues
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        self.session = aiohttp.ClientSession(connector=connector)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
    
    async def generate_response(self, model_config: ModelConfig, test_case: TestCase) -> str:
        """Generate response from AI model via OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-org/benchmark1",  # Replace with your repo
            "X-Title": "FHIR AI Benchmark"
        }
        
        messages = []
        
        # Add system prompt if provided
        if test_case.system_prompt:
            messages.append({
                "role": "system", 
                "content": test_case.system_prompt
            })
        
        # Add user prompt
        messages.append({
            "role": test_case.role,
            "content": test_case.prompt
        })
        
        payload = {
            "model": model_config.model_id,
            "messages": messages,
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature,
            "top_p": model_config.top_p
        }
        
        async with self.session.post(self.base_url, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()
                return data["choices"][0]["message"]["content"]
            else:
                error_text = await response.text()
                raise Exception(f"OpenRouter API call failed: {response.status} - {error_text}")

class BenchmarkRunner:
    """Main benchmark execution engine."""
    
    def __init__(self, tests_dir: str, models_file: str, openrouter_api_key: str, output_dir: str = "results"):
        self.tests_dir = Path(tests_dir)
        self.test_cases = self._load_test_cases()
        self.models = self._load_models(models_file)
        self.openrouter_client = None
        self.openrouter_api_key = openrouter_api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
    
    def _load_test_cases(self) -> List[TestCase]:
        """Load test cases from individual YAML files."""
        test_cases = []
        
        for yaml_file in self.tests_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                
                # Use filename (without extension) as ID
                test_id = yaml_file.stem
                
                # Set default scoring criteria if not provided
                if 'scoring_criteria' not in data:
                    data['scoring_criteria'] = {
                        'fhir_validity': 0.4,
                        'resource_type': 0.2,
                        'error_severity': 0.2,
                        'completeness': 0.2
                    }
                
                test_case = TestCase(id=test_id, **data)
                test_cases.append(test_case)
                logger.info(f"Loaded test case: {test_id}")
                
            except Exception as e:
                logger.error(f"Error loading test case {yaml_file}: {e}")
        
        logger.info(f"Loaded {len(test_cases)} test cases")
        return test_cases
    
    def _load_models(self, file_path: str) -> List[ModelConfig]:
        """Load model configurations from YAML file."""
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return [ModelConfig(**model) for model in data['models']]
    
    async def run_benchmark(self):
        """Run the complete benchmark across all models and test cases."""
        logger.info(f"Starting benchmark with {len(self.test_cases)} tests and {len(self.models)} models")
        
        async with OpenRouterClient(self.openrouter_api_key) as openrouter_client:
            async with FHIRValidator() as validator:
                
                for model_config in self.models:
                    logger.info(f"Testing model: {model_config.name} ({model_config.model_id})")
                    
                    for test_case in self.test_cases:
                        try:
                            result = await self._run_single_test(
                                openrouter_client, model_config, test_case, validator
                            )
                            self.results.append(result)
                            logger.info(f"✓ {model_config.name} - {test_case.id}: {result.total_score:.2f}")
                            
                            # Add small delay to respect rate limits
                            await asyncio.sleep(1)
                            
                        except Exception as e:
                            logger.error(f"✗ {model_config.name} - {test_case.id}: {e}")
                            error_result = TestResult(
                                test_id=test_case.id,
                                model_name=model_config.name,
                                response="",
                                fhir_validation={},
                                scores={},
                                total_score=0.0,
                                execution_time=0.0,
                                timestamp=datetime.now(),
                                error=str(e)
                            )
                            self.results.append(error_result)
        
        # Generate reports
        await self._generate_reports()
    
    async def _run_single_test(self, client: OpenRouterClient, model_config: ModelConfig, 
                               test_case: TestCase, validator: FHIRValidator) -> TestResult:
        """Run a single test case against a model."""
        start_time = time.time()
        
        # Get AI response via OpenRouter
        response = await client.generate_response(model_config, test_case)
        
        # Extract JSON from response (AI might add explanations)
        json_response = self._extract_json(response)
        
        # Validate FHIR
        validation_result = await validator.validate_resource(json_response)
        
        # Calculate scores
        scores = self._calculate_scores(test_case, json_response, validation_result)
        total_score = sum(score * weight for score, weight in 
                         zip(scores.values(), test_case.scoring_criteria.values()))
        
        execution_time = time.time() - start_time
        
        return TestResult(
            test_id=test_case.id,
            model_name=model_config.name,
            response=response,
            fhir_validation=validation_result,
            scores=scores,
            total_score=total_score,
            execution_time=execution_time,
            timestamp=datetime.now()
        )
    
    def _extract_json(self, response: str) -> str:
        """Extract JSON from AI response, handling cases where AI adds explanations."""
        # Try to find JSON in the response
        response = response.strip()
        
        # Look for JSON block markers
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end != -1:
                return response[start:end].strip()
        
        # Look for JSON object starting with {
        start = response.find("{")
        if start != -1:
            # Simple approach: find the matching closing brace
            brace_count = 0
            for i, char in enumerate(response[start:]):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return response[start:start + i + 1]
        
        # If no JSON found, return the original response
        return response
    
    def _calculate_scores(self, test_case: TestCase, response: str, 
                         validation: Dict[str, Any]) -> Dict[str, float]:
        """Calculate individual scores for different criteria."""
        scores = {}
        
        # FHIR validity score (0 or 1)
        scores['fhir_validity'] = 1.0 if validation['valid'] else 0.0
        
        # Resource type correctness (if specified)
        if test_case.expected_resource_type:
            expected_type = test_case.expected_resource_type
            actual_type = validation.get('resource_type', 'unknown')
            scores['resource_type'] = 1.0 if actual_type == expected_type else 0.0
        else:
            scores['resource_type'] = 1.0  # No requirement specified
        
        # Error severity scoring
        error_count = len(validation.get('errors', []))
        warning_count = len(validation.get('warnings', []))
        scores['error_severity'] = max(0.0, 1.0 - (error_count * 0.2 + warning_count * 0.1))
        
        # Response completeness (basic heuristic based on JSON structure)
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                # Score based on presence of key FHIR fields
                required_fields = ['resourceType', 'id']
                present_fields = sum(1 for field in required_fields if field in parsed)
                field_score = present_fields / len(required_fields)
                
                # Bonus for additional meaningful content
                total_fields = len(parsed.keys())
                content_score = min(1.0, total_fields / 8.0)  # Normalize to reasonable field count
                
                scores['completeness'] = (field_score + content_score) / 2
            else:
                scores['completeness'] = 0.0
        except:
            scores['completeness'] = 0.0
        
        return scores
    
    async def _generate_reports(self):
        """Generate HTML and JSON reports."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw results as JSON
        json_output = self.output_dir / f"results_{timestamp}.json"
        with open(json_output, 'w') as f:
            json.dump([asdict(result) for result in self.results], f, indent=2, default=str)
        
        # Generate summary statistics
        summary = self._generate_summary()
        
        # Save summary
        summary_output = self.output_dir / f"summary_{timestamp}.json"
        with open(summary_output, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generate HTML report
        html_output = self.output_dir / "benchmark_report.html"
        await self._generate_html_report(html_output, summary)
        
        # Generate latest symlinks for easy access
        latest_json = self.output_dir / "latest_results.json"
        latest_summary = self.output_dir / "latest_summary.json"
        
        if latest_json.exists():
            latest_json.unlink()
        if latest_summary.exists():
            latest_summary.unlink()
            
        latest_json.symlink_to(json_output.name)
        latest_summary.symlink_to(summary_output.name)
        
        logger.info(f"Reports generated: {html_output}")
        logger.info(f"Raw results: {json_output}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            "run_timestamp": datetime.now().isoformat(),
            "total_tests": len(self.results),
            "models": {},
            "test_categories": {},
            "overall_stats": {}
        }
        
        # Per-model statistics
        for model_name in set(r.model_name for r in self.results):
            model_results = [r for r in self.results if r.model_name == model_name]
            successful_results = [r for r in model_results if r.error is None]
            
            if successful_results:
                summary["models"][model_name] = {
                    "avg_score": sum(r.total_score for r in successful_results) / len(successful_results),
                    "success_rate": len(successful_results) / len(model_results),
                    "avg_execution_time": sum(r.execution_time for r in successful_results) / len(successful_results),
                    "fhir_validity_rate": sum(1 for r in successful_results if r.fhir_validation.get('valid', False)) / len(successful_results),
                    "test_count": len(model_results),
                    "error_count": len(model_results) - len(successful_results)
                }
            else:
                summary["models"][model_name] = {
                    "avg_score": 0.0,
                    "success_rate": 0.0,
                    "avg_execution_time": 0.0,
                    "fhir_validity_rate": 0.0,
                    "test_count": len(model_results),
                    "error_count": len(model_results)
                }
        
        # Per-test-case statistics
        for test_id in set(r.test_id for r in self.results):
            test_results = [r for r in self.results if r.test_id == test_id]
            successful_results = [r for r in test_results if r.error is None]
            
            if successful_results:
                summary["test_categories"][test_id] = {
                    "avg_score": sum(r.total_score for r in successful_results) / len(successful_results),
                    "model_count": len(successful_results),
                    "fhir_validity_rate": sum(1 for r in successful_results if r.fhir_validation.get('valid', False)) / len(successful_results)
                }
        
        return summary
    
    async def _generate_html_report(self, output_path: Path, summary: Dict[str, Any]):
        """Generate HTML report with charts and tables."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>FHIR AI Benchmark Results</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               margin: 0; padding: 40px; background: #f8f9fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; 
                     padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ background: #e3f2fd; padding: 25px; border-radius: 8px; margin-bottom: 30px; }}
        .model-card {{ border: 1px solid #e0e0e0; margin: 20px 0; padding: 20px; 
                       border-radius: 8px; background: #fafafa; }}
        .score {{ font-weight: bold; }}
        .score.excellent {{ color: #2e7d32; }}
        .score.good {{ color: #1976d2; }}
        .score.poor {{ color: #d32f2f; }}
        .error {{ color: #d32f2f; background: #ffebee; padding: 8px; border-radius: 4px; }}
        .success {{ color: #2e7d32; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: white; }}
        th, td {{ border: 1px solid #e0e0e0; padding: 12px; text-align: left; }}
        th {{ background-color: #f5f5f5; font-weight: 600; }}
        .metric {{ display: inline-block; margin: 10px 20px 10px 0; }}
        .metric-value {{ font-size: 1.5em; font-weight: bold; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏥 FHIR AI Benchmark Results</h1>
            <p><strong>Generated:</strong> {summary['run_timestamp']}</p>
            <p><strong>Total Tests:</strong> {summary['total_tests']} | <strong>Models:</strong> {len(summary['models'])}</p>
        </div>
        
        <h2>📊 Model Performance Leaderboard</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Model</th>
                <th>Overall Score</th>
                <th>FHIR Validity Rate</th>
                <th>Success Rate</th>
                <th>Avg Time (s)</th>
                <th>Tests Completed</th>
            </tr>
"""
        
        # Sort models by average score
        sorted_models = sorted(summary["models"].items(), 
                              key=lambda x: x[1]['avg_score'], reverse=True)
        
        for rank, (model_name, stats) in enumerate(sorted_models, 1):
            score_class = "excellent" if stats['avg_score'] > 0.8 else "good" if stats['avg_score'] > 0.5 else "poor"
            html_content += f"""
            <tr>
                <td><strong>{rank}</strong></td>
                <td>{model_name}</td>
                <td><span class="score {score_class}">{stats['avg_score']:.3f}</span></td>
                <td class="{'success' if stats['fhir_validity_rate'] > 0.8 else 'error'}">{stats['fhir_validity_rate']:.1%}</td>
                <td class="{'success' if stats['success_rate'] > 0.8 else 'error'}">{stats['success_rate']:.1%}</td>
                <td>{stats['avg_execution_time']:.2f}</td>
                <td>{stats['test_count']}</td>
            </tr>
"""
        
        html_content += """
        </table>
        
        <h2>🧪 Test Case Performance</h2>
        <table>
            <tr>
                <th>Test Case</th>
                <th>Average Score</th>
                <th>FHIR Validity Rate</th>
                <th>Models Tested</th>
            </tr>
"""
        
        for test_id, stats in summary["test_categories"].items():
            html_content += f"""
            <tr>
                <td><strong>{test_id}</strong></td>
                <td class="score">{stats['avg_score']:.3f}</td>
                <td class="{'success' if stats['fhir_validity_rate'] > 0.8 else 'error'}">{stats['fhir_validity_rate']:.1%}</td>
                <td>{stats['model_count']}</td>
            </tr>
"""
        
        html_content += """
        </table>
        
        <h2>📝 Detailed Results by Model</h2>
"""
        
        # Detailed results for each model
        for model_name in summary["models"].keys():
            model_results = [r for r in self.results if r.model_name == model_name]
            stats = summary["models"][model_name]
            
            html_content += f"""
        <div class="model-card">
            <h3>🤖 {model_name}</h3>
            <div class="metric">
                <div class="metric-value score">{stats['avg_score']:.3f}</div>
                <div class="metric-label">Average Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{stats['fhir_validity_rate']:.1%}</div>
                <div class="metric-label">FHIR Valid</div>
            </div>
            <div class="metric">
                <div class="metric-value">{stats['avg_execution_time']:.2f}s</div>
                <div class="metric-label">Avg Time</div>
            </div>
            
            <table>
                <tr>
                    <th>Test Case</th>
                    <th>Score</th>
                    <th>FHIR Valid</th>
                    <th>Errors</th>
                    <th>Status</th>
                </tr>
"""
            
            for result in model_results:
                if result.error:
                    status_cell = f'<span class="error">Error: {result.error}</span>'
                    valid_icon = "❌"
                    error_count = "N/A"
                else:
                    status_cell = '<span class="success">✓ Success</span>'
                    valid_icon = "✅" if result.fhir_validation.get('valid', False) else "❌"
                    error_count = len(result.fhir_validation.get('errors', []))
                
                html_content += f"""
                <tr>
                    <td><strong>{result.test_id}</strong></td>
                    <td class="score">{result.total_score:.3f}</td>
                    <td>{valid_icon}</td>
                    <td>{error_count}</td>
                    <td>{status_cell}</td>
                </tr>
"""
            
            html_content += "</table></div>"
        
        html_content += """
        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #666;">
            <p>Generated by FHIR AI Benchmark Framework</p>
        </footer>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)

async def main():
    """Main execution function."""
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(
        tests_dir="tests",
        models_file="models.yaml",
        openrouter_api_key=openrouter_api_key,
        output_dir="results"
    )
    
    # Run benchmark
    await runner.run_benchmark()
    
    print("Benchmark completed! Check the results directory for reports.")

if __name__ == "__main__":
    asyncio.run(main())