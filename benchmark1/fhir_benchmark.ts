#!/usr/bin/env node

/**
 * FHIR AI Benchmark Framework - TypeScript Version
 * A tool for evaluating AI models on FHIR resource generation tasks using OpenRouter.
 * Uses fhir-validator-wrapper for FHIR validation.
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import yaml from 'yaml';
import fetch from 'node-fetch';
import FhirValidator from 'fhir-validator-wrapper';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Type definitions
interface TestCase {
  id: string;
  name: string;
  role: string;
  system_prompt: string;
  prompt: string;
  expected_resource_type?: string;
  profile?: string; // FHIR profile to validate against
  category: string;
  difficulty: string;
  scoring_criteria: Record<string, number>;
  metadata?: Record<string, any>;
}

interface ModelConfig {
  name: string;
  model_id: string; // OpenRouter model ID (e.g., "anthropic/claude-3.5-sonnet")
  max_tokens: number;
  temperature: number;
  top_p: number;
}

interface TestResult {
  test_id: string;
  model_name: string;
  response: string;
  fhir_validation: FHIRValidationResult;
  scores: Record<string, number>;
  total_score: number;
  execution_time: number;
  timestamp: Date;
  error?: string;
}

interface FHIRValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  resource_type: string;
  issues?: any[]; // Full validation issues from OperationOutcome
}

interface ValidationConfig {
  validatorJarPath: string;
  version: string;
  txServer: string;
  txLog: string;
  igs?: string[];
  port?: number;
  timeout?: number;
}

class Logger {
  private static formatTime(): string {
    return new Date().toISOString();
  }

  static info(message: string): void {
    console.log(`${this.formatTime()} - INFO - ${message}`);
  }

  static error(message: string): void {
    console.error(`${this.formatTime()} - ERROR - ${message}`);
  }
}

class OpenRouterClient {
  private baseUrl = "https://openrouter.ai/api/v1/chat/completions";

  constructor(private apiKey: string) {}

  async generateResponse(modelConfig: ModelConfig, testCase: TestCase): Promise<string> {
    const headers = {
      "Authorization": `Bearer ${this.apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer": "https://github.com/your-org/benchmark1",
      "X-Title": "FHIR AI Benchmark"
    };

    const messages: Array<{ role: string; content: string }> = [];

    // Add system prompt if provided
    if (testCase.system_prompt) {
      messages.push({
        role: "system",
        content: testCase.system_prompt
      });
    }

    // Add user prompt
    messages.push({
      role: testCase.role,
      content: testCase.prompt
    });

    const payload = {
      model: modelConfig.model_id,
      messages,
      max_tokens: modelConfig.max_tokens,
      temperature: modelConfig.temperature,
      top_p: modelConfig.top_p
    };

    const response = await fetch(this.baseUrl, {
      method: 'POST',
      headers,
      body: JSON.stringify(payload)
    });

    if (response.ok) {
      const data = await response.json() as any;
      return data.choices[0].message.content;
    } else {
      const errorText = await response.text();
      throw new Error(`OpenRouter API call failed: ${response.status} - ${errorText}`);
    }
  }
}

class BenchmarkRunner {
  private testCases: TestCase[];
  private models: ModelConfig[];
  private outputDir: string;
  private jsonOutputDir: string;
  private results: TestResult[] = [];
  private validationConfig: ValidationConfig;
  private validator: FhirValidator;

  constructor(
      private testsDir: string,
      private modelsFile: string,
      private validationConfigFile: string,
      private openrouterApiKey: string,
      outputDir: string = "results"
  ) {
    this.outputDir = path.resolve(outputDir);
    this.jsonOutputDir = path.join(this.outputDir, "output");
  }

  async initialize(): Promise<void> {
    // Create output directories
    await fs.mkdir(this.outputDir, { recursive: true });
    await fs.mkdir(this.jsonOutputDir, { recursive: true });

    // Load configurations
    this.testCases = await this.loadTestCases();
    this.models = await this.loadModels();
    this.validationConfig = await this.loadValidationConfig();

    // Initialize FHIR validator
    this.validator = new FhirValidator(this.validationConfig.validatorJarPath);

    Logger.info("Starting FHIR validator service...");
    await this.validator.start({
      version: this.validationConfig.version,
      txServer: this.validationConfig.txServer,
      txLog: this.validationConfig.txLog,
      igs: this.validationConfig.igs || [],
      port: this.validationConfig.port || 8080,
      timeout: this.validationConfig.timeout || 30000
    });
    Logger.info("FHIR validator service started successfully");
  }

  async cleanup(): Promise<void> {
    if (this.validator) {
      Logger.info("Stopping FHIR validator service...");
      await this.validator.stop();
    }
  }

  private async loadTestCases(): Promise<TestCase[]> {
    const testCases: TestCase[] = [];

    try {
      const files = await fs.readdir(this.testsDir);
      const yamlFiles = files.filter(file => file.endsWith('.yaml') || file.endsWith('.yml'));

      for (const yamlFile of yamlFiles) {
        try {
          const filePath = path.join(this.testsDir, yamlFile);
          const content = await fs.readFile(filePath, 'utf-8');
          const data = yaml.parse(content);

          // Use filename (without extension) as ID
          const testId = path.basename(yamlFile, path.extname(yamlFile));

          // Set default scoring criteria if not provided
          if (!data.scoring_criteria) {
            data.scoring_criteria = {
              fhir_validity: 0.4,
              resource_type: 0.2,
              error_severity: 0.2,
              completeness: 0.2
            };
          }

          const testCase: TestCase = {
            id: testId,
            ...data
          };

          testCases.push(testCase);
          Logger.info(`Loaded test case: ${testId}`);

        } catch (error) {
          Logger.error(`Error loading test case ${yamlFile}: ${error}`);
        }
      }
    } catch (error) {
      Logger.error(`Error reading tests directory: ${error}`);
    }

    Logger.info(`Loaded ${testCases.length} test cases`);
    return testCases;
  }

  private async loadModels(): Promise<ModelConfig[]> {
    const content = await fs.readFile(this.modelsFile, 'utf-8');
    const data = yaml.parse(content);
    return data.models.map((model: any) => ({
      max_tokens: 4000,
      temperature: 0.1,
      top_p: 1.0,
      ...model
    }));
  }

  private async loadValidationConfig(): Promise<ValidationConfig> {
    const content = await fs.readFile(this.validationConfigFile, 'utf-8');
    const data = yaml.parse(content);

    return {
      port: 8080,
      timeout: 30000,
      ...data
    };
  }

  async runBenchmark(): Promise<void> {
    Logger.info(`Starting benchmark with ${this.testCases.length} tests and ${this.models.length} models`);

    const openrouterClient = new OpenRouterClient(this.openrouterApiKey);

    try {
      for (const modelConfig of this.models) {
        Logger.info(`Testing model: ${modelConfig.name} (${modelConfig.model_id})`);

        for (const testCase of this.testCases) {
          try {
            const result = await this.runSingleTest(
                openrouterClient,
                modelConfig,
                testCase
            );
            this.results.push(result);
            Logger.info(`✅ ${modelConfig.name} - ${testCase.id}: ${result.total_score.toFixed(2)}`);

            // Add small delay to respect rate limits
            await new Promise(resolve => setTimeout(resolve, 1000));

          } catch (error) {
            Logger.error(`❌ ${modelConfig.name} - ${testCase.id}: ${error}`);
            const errorResult: TestResult = {
              test_id: testCase.id,
              model_name: modelConfig.name,
              response: "",
              fhir_validation: {
                valid: false,
                errors: [String(error)],
                warnings: [],
                resource_type: 'unknown'
              },
              scores: {},
              total_score: 0.0,
              execution_time: 0.0,
              timestamp: new Date(),
              error: String(error)
            };
            this.results.push(errorResult);
          }
        }
      }
    } finally {
      await this.cleanup();
    }

    // Generate reports
    await this.generateReports();
  }

  private async runSingleTest(
      client: OpenRouterClient,
      modelConfig: ModelConfig,
      testCase: TestCase
  ): Promise<TestResult> {
    const startTime = Date.now();

    // Get AI response via OpenRouter
    const response = await client.generateResponse(modelConfig, testCase);

    // Save response to JSON file
    const jsonFilename = `${testCase.id}_${modelConfig.name}_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
    const jsonFilepath = path.join(this.jsonOutputDir, jsonFilename);
    await fs.writeFile(jsonFilepath, response);

    // Extract JSON from response (AI might add explanations)
    const jsonResponse = this.extractJson(response);

    // Validate FHIR using fhir-validator-wrapper
    const validationResult = await this.validateFhirResource(jsonResponse, testCase.profile);

    // Calculate scores
    const scores = this.calculateScores(testCase, jsonResponse, validationResult);
    const totalScore = Object.entries(scores).reduce((total, [key, score]) => {
      const weight = testCase.scoring_criteria[key] || 0;
      return total + (score * weight);
    }, 0);

    const executionTime = (Date.now() - startTime) / 1000;

    return {
      test_id: testCase.id,
      model_name: modelConfig.name,
      response,
      fhir_validation: validationResult,
      scores,
      total_score: totalScore,
      execution_time: executionTime,
      timestamp: new Date()
    };
  }

  private async validateFhirResource(fhirJson: string, profile?: string): Promise<FHIRValidationResult> {
    try {
      // Parse JSON to ensure it's valid and get resource type
      const resource = JSON.parse(fhirJson);
      const resourceType = resource?.resourceType || 'unknown';

      // Build validation options for the fhir-validator-wrapper
      const validationOptions: any = {
        anyExtensionsAllowed: true
      };

      // Add profile if specified
      if (profile) {
        validationOptions.profiles = [profile];
      }

      // Call the fhir-validator-wrapper's validate method
      const operationOutcome = await this.validator.validate(fhirJson, validationOptions);

      // Process OperationOutcome to extract validation info
      const issues = operationOutcome?.issue || [];
      const errors: string[] = [];
      const warnings: string[] = [];

      for (const issue of issues) {
        const message = issue.diagnostics || issue.details?.text || 'Unknown issue';

        if (issue.severity === 'error' || issue.severity === 'fatal') {
          errors.push(message);
        } else if (issue.severity === 'warning') {
          warnings.push(message);
        }
      }

      const valid = errors.length === 0;

      return {
        valid,
        errors,
        warnings,
        resource_type: resourceType,
        issues
      };

    } catch (e) {
      if (e instanceof SyntaxError) {
        return {
          valid: false,
          errors: [`Invalid JSON: ${e.message}`],
          warnings: [],
          resource_type: 'unknown'
        };
      }

      Logger.error(`Validation error: ${e}`);
      return {
        valid: false,
        errors: [`Validation failed: ${e}`],
        warnings: [],
        resource_type: 'unknown'
      };
    }
  }

  private extractJson(response: string): string {
    // Try to find JSON in the response
    response = response.trim();

    // Look for JSON block markers
    if (response.includes("```json")) {
      const start = response.indexOf("```json") + 7;
      const end = response.indexOf("```", start);
      if (end !== -1) {
        return response.substring(start, end).trim();
      }
    }

    // Look for JSON object starting with {
    const start = response.indexOf("{");
    if (start !== -1) {
      // Simple approach: find the matching closing brace
      let braceCount = 0;
      for (let i = start; i < response.length; i++) {
        if (response[i] === "{") {
          braceCount++;
        } else if (response[i] === "}") {
          braceCount--;
          if (braceCount === 0) {
            return response.substring(start, i + 1);
          }
        }
      }
    }

    // If no JSON found, return the original response
    return response;
  }

  private calculateScores(
      testCase: TestCase,
      response: string,
      validation: FHIRValidationResult
  ): Record<string, number> {
    const scores: Record<string, number> = {};

    // FHIR validity score (0 or 1)
    scores.fhir_validity = validation.valid ? 1.0 : 0.0;

    // Resource type correctness (if specified)
    if (testCase.expected_resource_type) {
      const expectedType = testCase.expected_resource_type;
      const actualType = validation.resource_type || 'unknown';
      scores.resource_type = actualType === expectedType ? 1.0 : 0.0;
    } else {
      scores.resource_type = 1.0; // No requirement specified
    }

    // Error severity scoring
    const errorCount = validation.errors.length;
    const warningCount = validation.warnings.length;
    scores.error_severity = Math.max(0.0, 1.0 - (errorCount * 0.2 + warningCount * 0.1));

    // Response completeness (basic heuristic based on JSON structure)
    try {
      const parsed = JSON.parse(response);
      if (typeof parsed === 'object' && parsed !== null) {
        // Score based on presence of key FHIR fields
        const requiredFields = ['resourceType', 'id'];
        const presentFields = requiredFields.filter(field => field in parsed).length;
        const fieldScore = presentFields / requiredFields.length;

        // Bonus for additional meaningful content
        const totalFields = Object.keys(parsed).length;
        const contentScore = Math.min(1.0, totalFields / 8.0); // Normalize to reasonable field count

        scores.completeness = (fieldScore + contentScore) / 2;
      } else {
        scores.completeness = 0.0;
      }
    } catch {
      scores.completeness = 0.0;
    }

    return scores;
  }

  private async generateReports(): Promise<void> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 15);

    // Save raw results as JSON
    const jsonOutput = path.join(this.outputDir, `results_${timestamp}.json`);
    await fs.writeFile(jsonOutput, JSON.stringify(this.results, null, 2));

    // Generate summary statistics
    const summary = this.generateSummary();

    // Save summary
    const summaryOutput = path.join(this.outputDir, `summary_${timestamp}.json`);
    await fs.writeFile(summaryOutput, JSON.stringify(summary, null, 2));

    // Generate HTML report
    const htmlOutput = path.join(this.outputDir, "benchmark_report.html");
    await this.generateHtmlReport(htmlOutput, summary);

    Logger.info(`Reports generated: ${htmlOutput}`);
    Logger.info(`Raw results: ${jsonOutput}`);
  }

  private generateSummary(): Record<string, any> {
    const summary = {
      run_timestamp: new Date().toISOString(),
      total_tests: this.results.length,
      models: {} as Record<string, any>,
      test_categories: {} as Record<string, any>,
      overall_stats: {}
    };

    // Per-model statistics
    const modelNames = [...new Set(this.results.map(r => r.model_name))];
    for (const modelName of modelNames) {
      const modelResults = this.results.filter(r => r.model_name === modelName);
      const successfulResults = modelResults.filter(r => !r.error);

      if (successfulResults.length > 0) {
        const avgScore = successfulResults.reduce((sum, r) => sum + r.total_score, 0) / successfulResults.length;
        const successRate = successfulResults.length / modelResults.length;
        const avgExecTime = successfulResults.reduce((sum, r) => sum + r.execution_time, 0) / successfulResults.length;
        const fhirValidRate = successfulResults.filter(r => r.fhir_validation?.valid).length / successfulResults.length;

        summary.models[modelName] = {
          avg_score: avgScore,
          success_rate: successRate,
          avg_execution_time: avgExecTime,
          fhir_validity_rate: fhirValidRate,
          test_count: modelResults.length,
          error_count: modelResults.length - successfulResults.length
        };
      } else {
        summary.models[modelName] = {
          avg_score: 0.0,
          success_rate: 0.0,
          avg_execution_time: 0.0,
          fhir_validity_rate: 0.0,
          test_count: modelResults.length,
          error_count: modelResults.length
        };
      }
    }

    // Per-test-case statistics
    const testIds = [...new Set(this.results.map(r => r.test_id))];
    for (const testId of testIds) {
      const testResults = this.results.filter(r => r.test_id === testId);
      const successfulResults = testResults.filter(r => !r.error);

      if (successfulResults.length > 0) {
        const avgScore = successfulResults.reduce((sum, r) => sum + r.total_score, 0) / successfulResults.length;
        const fhirValidRate = successfulResults.filter(r => r.fhir_validation?.valid).length / successfulResults.length;

        summary.test_categories[testId] = {
          avg_score: avgScore,
          model_count: successfulResults.length,
          fhir_validity_rate: fhirValidRate
        };
      }
    }

    return summary;
  }

  private async generateHtmlReport(outputPath: string, summary: Record<string, any>): Promise<void> {
    const htmlContent = `
<!DOCTYPE html>
<html>
<head>
    <title>FHIR AI Benchmark Results</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               margin: 0; padding: 40px; background: #f8f9fa; }
        .container { max-width: 1200px; margin: 0 auto; background: white; 
                     padding: 30px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { background: #e3f2fd; padding: 25px; border-radius: 8px; margin-bottom: 30px; }
        .model-card { border: 1px solid #e0e0e0; margin: 20px 0; padding: 20px; 
                       border-radius: 8px; background: #fafafa; }
        .score { font-weight: bold; }
        .score.excellent { color: #2e7d32; }
        .score.good { color: #1976d2; }
        .score.poor { color: #d32f2f; }
        .error { color: #d32f2f; background: #ffebee; padding: 8px; border-radius: 4px; }
        .success { color: #2e7d32; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; background: white; }
        th, td { border: 1px solid #e0e0e0; padding: 12px; text-align: left; }
        th { background-color: #f5f5f5; font-weight: 600; }
        .metric { display: inline-block; margin: 10px 20px 10px 0; }
        .metric-value { font-size: 1.5em; font-weight: bold; }
        .metric-label { color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>FHIR AI Benchmark Results</h1>
            <p><strong>Generated:</strong> ${summary.run_timestamp}</p>
            <p><strong>Total Tests:</strong> ${summary.total_tests} | <strong>Models:</strong> ${Object.keys(summary.models).length}</p>
        </div>
        
        <h2>Model Performance Leaderboard</h2>
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
${this.generateModelTable(summary)}
        </table>
        
        <h2>Test Case Performance</h2>
        <table>
            <tr>
                <th>Test Case</th>
                <th>Average Score</th>
                <th>FHIR Validity Rate</th>
                <th>Models Tested</th>
            </tr>
${this.generateTestCaseTable(summary)}
        </table>
        
        <h2>Detailed Results by Model</h2>
${this.generateDetailedResults(summary)}
        
        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #666;">
            <p>Generated by FHIR AI Benchmark Framework</p>
        </footer>
    </div>
</body>
</html>
`;

    await fs.writeFile(outputPath, htmlContent);
  }

  private generateModelTable(summary: Record<string, any>): string {
    const sortedModels = Object.entries(summary.models as Record<string, any>)
        .sort(([, a], [, b]) => (b as any).avg_score - (a as any).avg_score);

    return sortedModels.map(([modelName, stats]: [string, any], index) => {
      const scoreClass = stats.avg_score > 0.8 ? "excellent" : stats.avg_score > 0.5 ? "good" : "poor";
      const validityClass = stats.fhir_validity_rate > 0.8 ? "success" : "error";
      const successClass = stats.success_rate > 0.8 ? "success" : "error";

      return `
            <tr>
                <td><strong>${index + 1}</strong></td>
                <td>${modelName}</td>
                <td><span class="score ${scoreClass}">${stats.avg_score.toFixed(3)}</span></td>
                <td class="${validityClass}">${(stats.fhir_validity_rate * 100).toFixed(1)}%</td>
                <td class="${successClass}">${(stats.success_rate * 100).toFixed(1)}%</td>
                <td>${stats.avg_execution_time.toFixed(2)}</td>
                <td>${stats.test_count}</td>
            </tr>`;
    }).join('');
  }

  private generateTestCaseTable(summary: Record<string, any>): string {
    return Object.entries(summary.test_categories as Record<string, any>).map(([testId, stats]: [string, any]) => {
      const validityClass = stats.fhir_validity_rate > 0.8 ? "success" : "error";

      return `
            <tr>
                <td><strong>${testId}</strong></td>
                <td class="score">${stats.avg_score.toFixed(3)}</td>
                <td class="${validityClass}">${(stats.fhir_validity_rate * 100).toFixed(1)}%</td>
                <td>${stats.model_count}</td>
            </tr>`;
    }).join('');
  }

  private generateDetailedResults(summary: Record<string, any>): string {
    return Object.keys(summary.models).map(modelName => {
      const modelResults = this.results.filter(r => r.model_name === modelName);
      const stats = summary.models[modelName];

      const tableRows = modelResults.map(result => {
        let statusCell: string;
        let validIcon: string;
        let errorCount: string;

        if (result.error) {
          statusCell = `<span class="error">Error: ${result.error}</span>`;
          validIcon = "❌";
          errorCount = "N/A";
        } else {
          statusCell = '<span class="success">✅ Success</span>';
          validIcon = result.fhir_validation?.valid ? "✅" : "❌";
          errorCount = String(result.fhir_validation?.errors?.length || 0);
        }

        return `
                <tr>
                    <td><strong>${result.test_id}</strong></td>
                    <td class="score">${result.total_score.toFixed(3)}</td>
                    <td>${validIcon}</td>
                    <td>${errorCount}</td>
                    <td>${statusCell}</td>
                </tr>`;
      }).join('');

      return `
        <div class="model-card">
            <h3>${modelName}</h3>
            <div class="metric">
                <div class="metric-value score">${stats.avg_score.toFixed(3)}</div>
                <div class="metric-label">Average Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">${(stats.fhir_validity_rate * 100).toFixed(1)}%</div>
                <div class="metric-label">FHIR Valid</div>
            </div>
            <div class="metric">
                <div class="metric-value">${stats.avg_execution_time.toFixed(2)}s</div>
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
${tableRows}
            </table>
        </div>`;
    }).join('');
  }
}

async function main(): Promise<void> {
  try {
    // Load environment variables - try multiple approaches
    try {
      const { config } = await import('dotenv');
      config();
    } catch (dotenvError) {
      Logger.info("dotenv not available, checking process.env directly");
    }

    // Check for API key in multiple ways
    let openrouterApiKey = process.env.OPENROUTER_API_KEY;

    // If not found, try reading .env file manually
    if (!openrouterApiKey) {
      try {
        const envContent = await fs.readFile('.env', 'utf-8');
        const envLines = envContent.split('\n');
        for (const line of envLines) {
          if (line.startsWith('OPENROUTER_API_KEY=')) {
            openrouterApiKey = line.split('=')[1].trim();
            break;
          }
        }
      } catch (envError) {
        Logger.error("Could not read .env file");
      }
    }

    if (!openrouterApiKey) {
      throw new Error(`OPENROUTER_API_KEY environment variable not set. 
Current env keys: ${Object.keys(process.env).filter(k => k.includes('ROUTER')).join(', ')}
Please set the environment variable or create a .env file with:
OPENROUTER_API_KEY=your_api_key_here`);
    }

    Logger.info(`API key loaded: ${openrouterApiKey.substring(0, 12)}...`);

    // Initialize benchmark runner
    const runner = new BenchmarkRunner(
        "tests",
        "models.yaml",
        "validation.yaml",
        openrouterApiKey,
        "results"
    );

    await runner.initialize();

    // Run benchmark
    await runner.runBenchmark();

    console.log("Benchmark completed! Check the results directory for reports.");
  } catch (error) {
    Logger.error(`Failed to run benchmark: ${error}`);
    process.exit(1);
  }
}

// Run if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(error => {
    Logger.error(`Benchmark failed: ${error}`);
    process.exit(1);
  });
}