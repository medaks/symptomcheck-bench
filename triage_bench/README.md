# Triage Benchmark

A comprehensive benchmark for evaluating Large Language Model (LLM) performance on medical triage classification tasks.

## Overview

This benchmark evaluates how accurately LLMs can classify medical vignettes into appropriate triage categories:

- **Emergency (em)**: Call 911 or go directly to the emergency room
- **Non-Emergency (ne)**: See a doctor, but symptoms don't require immediate ER attention (within a week)
- **Self-care (sc)**: Let the health issue resolve on its own and reassess in a few days

## Features

- **Multiple LLM Support**: OpenAI models (GPT-4o, GPT-4.5, O1, O3, etc.) and DeepSeek models
- **Vignettes**: Semigran clinical vignettes
- **Statistical Analysis**: Paired model comparison with McNemar tests
- **Safety Metrics**: Tracks over-triage and under-triage tendencies
- **Reproducible Results**: Supports multiple runs for statistical significance

## Installation

From the project root directory:

```bash
# Install dependencies
pip install -r requirements/development.txt
pip install -e .

# Set API keys
export KEY_OPENAI="sk-..."     # For OpenAI models
export KEY_DEEPSEEK="..."      # For DeepSeek models
```

## Usage

### Basic Evaluation

```bash
cd triage_bench

# Run with default settings (DeepSeek chat, Semigran vignettes, 1 run)
python3 main.py

# Run with GPT-4o
python3 main.py --model gpt-4o

# Run with multiple passes for statistical robustness
python3 main.py --model gpt-4o --runs 5

python3 main.py --model gpt-4o --vignette_set semigran
```

### Available Models

**OpenAI Models:**
- `gpt-4o`
- `gpt-4.5-preview` 
- `o1`, `o1-mini`
- `o3`, `o3-mini`
- `o4-mini`

**DeepSeek Models:**
- `deepseek-chat` (default)
- `deepseek-reasoner`

### Available Vignette Sets

- `semigran`: 45 clinical vignettes from Semigran et al.

### Example Commands

```bash
# Compare GPT-4o vs GPT-4.5 with 3 runs each
python3 main.py --model gpt-4o --runs 3
python3 main.py --model gpt-4.5-preview --runs 3

# Test DeepSeek reasoning model
python3 main.py --model deepseek-reasoner --vignette_set semigran --runs 2
```

## Statistical Analysis

### Paired Model Comparison

Compare two models statistically using McNemar's test:

```bash
python3 paired_analysis.py results/model1_results.jsonl results/model2_results.jsonl
```

**Example Output:**
```
=== MODEL A ===
Overall accuracy: 80.00%  (36/45)
Accuracy by triage level:
  em: 93.33%  (14/15)
  ne: 60.00%  (9/15)
  sc: 86.67%  (13/15)
Safety (at‑or‑above correct urgency): 88.89%  (40/45)
Over‑triage inclination (among incorrect): 44.44%  (4/9)

=== MODEL B ===
Overall accuracy: 77.78%  (35/45)
Safety (at‑or‑above correct urgency): 95.56%  (43/45)

=== McNemar paired test ===
discordant pairs: A‑right/B‑wrong = 5, A‑wrong/B‑right = 4
p‑value: 1.0   (exact=True)
Accuracy difference (B − A): -2.22%
```

## Output Format

### Results Files

Results are saved as timestamped JSONL files in the `results/` directory:

```
results/20250703T134639_gpt-4o_semigran_triage.jsonl
```

Each line contains:
```json
{
  "run_id": 1,
  "case_id": 1,
  "true_urgency": "em",
  "llm_output": "em",
  "correct": true,
  "model": "gpt-4o"
}
```

### Evaluation Metrics

- **Overall Accuracy**: Percentage of correct triage classifications
- **Per-Level Accuracy**: Accuracy for each triage category (em/ne/sc)
- **Safety Rate**: Percentage of predictions at or above correct urgency level
- **Over-triage Rate**: Among incorrect predictions, percentage that over-triaged

## Vignette Format

Vignettes are stored as JSONL files with the following structure:

```json
{
  "urgency_level": "em",
  "correct_diagnosis": "Acute liver failure",
  "case_description": "A 48-year-old woman with altered mental status..."
}
```

## Safety Considerations

The benchmark uses a conservative safety framework:
- **Under-triage** (predicting lower urgency than correct) is considered more dangerous
- **Over-triage** (predicting higher urgency than correct) is safer but may strain resources
- Safety metrics help evaluate model reliability for real-world deployment

## Research Applications

This benchmark is useful for:
- Comparing LLM performance on medical triage tasks
- Evaluating safety vs. efficiency trade-offs
- Testing prompt engineering approaches for medical AI
- Validating model reliability before clinical deployment

## File Structure

```
triage_bench/
├── README.md              # This file
├── main.py                 # Main evaluation script
├── paired_analysis.py      # Statistical comparison tool
├── vignettes/             # Clinical vignette datasets
│   └── semigran_vignettes.jsonl
└── results/               # Output directory for results
    ├── medask_results_jul25/  # Benchmarking results from July 2025 study
    └── [timestamped results files]
```

## Published Results

The `results/medask_results_jul25/` folder contains comprehensive benchmarking results from our July 2025 evaluation study, where MedAsk was compared against leading LLMs including OpenAI's O3 and GPT-4.5. These results are described in detail in our blog post:

**[Medical AI Triage Accuracy 2025: MedAsk Beats OpenAI's O3 & GPT-4.5](https://medask.tech/blogs/medical-ai-triage-accuracy-2025-medask-beats-openais-o3-gpt-4-5/)**

This study demonstrates MedAsk's superior performance in medical triage accuracy across multiple evaluation metrics.

## Contributing

When adding new vignette sets or models:
1. Follow the existing JSONL format for vignettes
2. Add model support in the client factory section of `main.py`
3. Update this README with new options
4. Test with multiple runs to ensure reproducibility

## Citation

If you use this benchmark in research, please cite the associated publication and link to the MedAsk blog posts mentioned in the main project README.