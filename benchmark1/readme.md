# FHIR AI Benchmark

A comprehensive framework for evaluating AI models on FHIR (Fast Healthcare Interoperability Resources) generation tasks. Tests multiple AI models against healthcare data generation scenarios and validates outputs using the official FHIR validator.

## Quick Start

### Prerequisites

- Node.js 18+ 
- TypeScript
- FHIR Validator JAR file
- OpenRouter API key

### Installation

```bash
npm install yaml node-fetch fhir-validator-wrapper
```

### Configuration

1. **Create environment file:**
```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
OPENROUTER_API_KEY=your_key_here
```

2. **Configure models** (`models.yaml`):
```yaml
models:
  - name: "GPT-4"
    model_id: "openai/gpt-4"
    max_tokens: 4000
    temperature: 0.1
  - name: "Claude-3.5-Sonnet"
    model_id: "anthropic/claude-3.5-sonnet"
    max_tokens: 4000
    temperature: 0.1
```

3. **Configure FHIR validation** (`validation.yaml`):
```yaml
validatorJarPath: "/path/to/validator_cli.jar"
version: "4.0.1"
txServer: "http://tx.fhir.org"
txLog: "tx.log"
igs: []
port: 8080
timeout: 30000
```

### Running

```bash
npx tsx fhir_benchmark.ts
```

## Writing Test Cases

Test cases are individual YAML files in the `tests/` directory. Each file represents one test scenario.

### Test File Structure

**File:** `tests/patient-basic.yaml`
```yaml
role: user
name: "Basic Patient Resource"
system_prompt: |
  You are a FHIR expert. Generate valid FHIR resources according to the R4 specification. 
  Return only JSON, no explanations.
prompt: |
  Create a FHIR Patient resource for:
  - Name: Sarah Johnson
  - Date of birth: 1985-03-15
  - Gender: female
  - Phone: +61 3 9876 5432
  - Address: 123 Collins Street, Melbourne, VIC 3000, Australia
expected_resource_type: "Patient"
profile: "http://hl7.org/fhir/StructureDefinition/Patient"
category: "basic_resources"
difficulty: "easy"
scoring_criteria:
  fhir_validity: 0.5
  resource_type: 0.2
  error_severity: 0.2
  completeness: 0.1
```

### Required Fields

- **`role`**: Usually "user" 
- **`name`**: Descriptive name for the test
- **`system_prompt`**: Instructions for the AI model
- **`prompt`**: The actual test question/task
- **`category`**: Group tests by type (e.g., "patient_management", "observations")
- **`difficulty`**: "easy", "medium", or "hard"
- **`scoring_criteria`**: Weights for different scoring aspects (must sum to 1.0)

### Optional Fields

- **`expected_resource_type`**: FHIR resource type to validate against
- **`profile`**: Specific FHIR profile URL for validation
- **`metadata`**: Additional information about the test

### Scoring Criteria

The framework evaluates four aspects:

- **`fhir_validity`**: Whether the output passes FHIR validation (0.0-1.0)
- **`resource_type`**: Whether the correct FHIR resource type was generated (0.0-1.0)
- **`error_severity`**: Penalty for validation errors and warnings (0.0-1.0)
- **`completeness`**: How complete the generated resource is (0.0-1.0)

Weights must sum to 1.0. Common patterns:
- Strict validation: `fhir_validity: 0.6, resource_type: 0.2, error_severity: 0.1, completeness: 0.1`
- Balanced: `fhir_validity: 0.4, resource_type: 0.2, error_severity: 0.2, completeness: 0.2`
- Content-focused: `fhir_validity: 0.3, resource_type: 0.1, error_severity: 0.1, completeness: 0.5`

### Test Categories

Organize tests by healthcare domain:

- **`patient_management`**: Patient demographics, identifiers
- **`clinical_data`**: Observations, vital signs, lab results
- **`medication_management`**: Prescriptions, administration records
- **`care_coordination`**: Care plans, referrals, encounters
- **`terminology`**: Coding systems, value sets

### Complex Example

**File:** `tests/bundle-discharge-summary.yaml`
```yaml
role: user
name: "Hospital Discharge Summary Bundle"
system_prompt: |
  You are an expert in FHIR R4 and hospital systems. Create comprehensive, 
  clinically accurate FHIR resources that would be found in real hospital discharge summaries.
prompt: |
  Create a FHIR Bundle containing a complete discharge summary for:
  - Patient: Maria Santos, 67-year-old female
  - Admission: Acute myocardial infarction
  - Length of stay: 4 days
  - Discharge medications: Aspirin 100mg daily, Atorvastatin 40mg daily
  - Follow-up: Cardiology clinic in 2 weeks
  
  Include Patient, Encounter, Condition, MedicationRequest, and Appointment resources.
expected_resource_type: "Bundle"
profile: "http://hl7.org/fhir/StructureDefinition/Bundle"
category: "care_coordination"
difficulty: "hard"
scoring_criteria:
  fhir_validity: 0.3
  resource_type: 0.1
  error_severity: 0.2
  completeness: 0.4
metadata:
  clinical_complexity: "high"
  expected_resources: ["Patient", "Encounter", "Condition", "MedicationRequest", "Appointment"]
```

## Output Structure

Results are organized in timestamped directories:

```
results/
└── 2025-09-09T1538/
    ├── patient-basic_GPT-4-response.txt           # Raw AI response
    ├── patient-basic_GPT-4-validation.txt         # FHIR validation results
    ├── patient-basic_Claude-3.5-Sonnet-response.txt
    ├── patient-basic_Claude-3.5-Sonnet-validation.txt
    ├── benchmark_report.html                      # Interactive results
    ├── results_2025-09-09T1538.json              # Raw data
    └── summary_2025-09-09T1538.json              # Aggregated metrics
```

### Validation Output Format

Validation files show detailed FHIR compliance:

```
Validation Result: INVALID
Resource Type: Patient

[12,5] [error] [Patient.identifier]: Missing required element 'system'
[23,15] [warning] [Patient.extension[0]]: Extension not recognized
[45,8] [error] [Patient.birthDate]: Invalid date format
```

## Best Practices

### Writing Effective Tests

1. **Be specific**: Clear, unambiguous requirements
2. **Test edge cases**: Invalid data, missing fields, complex scenarios
3. **Use real clinical scenarios**: Base tests on actual healthcare workflows
4. **Progressive difficulty**: Start with simple resources, build to complex bundles
5. **Cover terminology**: Include coding systems (SNOMED CT, LOINC, ICD-10)

### Test Organization

- Group related tests in categories
- Use descriptive filenames (`patient-demographics-basic.yaml`)
- Document expected clinical scenarios in metadata
- Include both positive and negative test cases

### Scoring Guidelines

- **High `fhir_validity` weight**: For strict conformance testing
- **High `completeness` weight**: For comprehensive data generation
- **Balanced weights**: For general-purpose evaluation
- **Consider clinical context**: Some fields are more critical than others

## Troubleshooting

**No test cases loaded**: Check that `.yaml` files exist in `tests/` directory

**Validation errors**: Ensure FHIR validator JAR path is correct in `validation.yaml`

**API errors**: Verify OpenRouter API key is set correctly

**Unicode issues**: Check file encoding is UTF-8

**Timestamp directories**: Each run creates a new timestamped folder - this is expected behavior