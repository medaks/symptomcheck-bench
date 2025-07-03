# MedAsk Benchmarks

A comprehensive suite of medical AI benchmarks for evaluating Large Language Model (LLM) performance on clinical tasks.

## Overview

This repository contains multiple medical AI benchmarks developed by MedAsk to evaluate and compare the performance of LLMs on various medical tasks:

### 🩺 SymptomCheck Bench
An OSCE-style benchmark for evaluating diagnostic accuracy of LLM-based medical agents in symptom assessment conversations. The benchmark simulates medical consultations through a structured four-step process:
1. Initialization: Selection of a clinical vignette  
2. Dialogue: Simulated conversation between an LLM agent and patient
3. Diagnosis: Generation of top 5 differential diagnoses
4. Evaluation: Automated assessment of diagnostic accuracy

**📖 Blog Post:** [Introducing SymptomCheck Bench](https://medask.tech/blogs/introducing-symptomcheck-bench/)

### 🚨 Triage Bench  
A benchmark for evaluating LLM performance on medical triage classification tasks. Models classify clinical vignettes into three urgency levels:
- **Emergency (em)**: Requires immediate emergency room attention
- **Non-Emergency (ne)**: Needs medical evaluation within a week  
- **Self-care (sc)**: Can be managed with self-care and monitoring

**📖 Blog Post:** [Medical AI Triage Accuracy 2025: MedAsk Beats OpenAI's O3 & GPT-4.5](https://medask.tech/blogs/medical-ai-triage-accuracy-2025-medask-beats-openais-o3-gpt-4-5/)

## Published Research & Results

### ICD-10 Coding Accuracy
Results from our ICD-10 coding evaluation demonstrate MedAsk's superior accuracy in medical coding tasks. They can be found in the results folder of symptomcheck_bench
**📖 Read more:** [How MedAsk's Cognitive Architecture Improves ICD-10 Coding Accuracy](https://medask.tech/blogs/how-medasks-cognitive-architecture-improves-icd-10-coding-accuracy/)

## Repository Structure

```
medask-benchmarks/
├── README.md                    # This overview
├── medask/                      # Core supporting functions and LLM clients
│   ├── ummon/                   # LLM client implementations
│   ├── models/                  # Data models for communication
│   └── util/                    # Utility functions
├── symptomcheck_bench/          # Diagnostic accuracy benchmark
│   ├── README.md               # Detailed usage instructions
│   ├── main.py                 # Main evaluation script
│   ├── vignettes/              # Clinical vignettes
│   └── results/                # Evaluation results
└── triage_bench/               # Medical triage benchmark  
    ├── README.md               # Detailed usage instructions
    ├── main.py                 # Main evaluation script
    ├── paired_analysis.py      # Statistical comparison tool
    ├── vignettes/              # Clinical vignettes
    └── results/                # Triage evaluation results
        └── medask_results_jul25/  # July 2025 study results
```

## Quick Start

### Installation

```bash
# Create and activate environment
conda create -n medask-benchmarks python=3.12
conda activate medask-benchmarks

# Install dependencies
pip install -r requirements/development.txt
pip install -e .

# Set API keys
export KEY_OPENAI="sk-..."     # For OpenAI models
export KEY_DEEPSEEK="..."      # For DeepSeek models
```

### Running Benchmarks

**SymptomCheck Bench:**
```bash
cd symptomcheck_bench
python main.py --file=avey --doctor_llm=gpt-4o --num_vignettes=5
```

**Triage Bench:**
```bash
cd triage_bench  
python main.py --model gpt-4o --runs 3
```

For detailed usage instructions, see the README files in each benchmark directory.

## Supported Models

- **OpenAI**: GPT-4o, GPT-4.5, O1, O3 series
- **DeepSeek**: DeepSeek Chat, DeepSeek Reasoner
- **MedAsk**: Proprietary medical AI models

## Citation

If you use these benchmarks in your research, please cite the associated publications and reference the relevant MedAsk blog posts linked above.

