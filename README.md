# MatSciTableExtract

A tool for extracting structured materials science data from tables using Large Language Models.

## Overview

MatSciTableExtract is designed to extract structured data from materials science tables using multiple input methods:
- **Image-based extraction** (directly processing table images)
- **OCR-based extraction** (extracting text from images before processing)
- **Structured format extraction** (parsing tables in CSV format)
- **Structured format with captions** (contextual captions are added to the input)

### Extracted Information
The system extracts key materials science information, including:
- **Matrix name**
- **Filler name**
- **Composition information** (amount and type)
- **Particle surface treatment (PST) name**
- **Material properties** (values, units, and conditions)

## Usage

### 1. Image-Based Extraction
To process images of tables, run:
```bash
python code/imagesasinput.py
```
This will:
- Process PNG images from the `tables` directory inside the articles in the data folder.
- Generate JSON output in `data/imageoutput3`.

### 2. Structured Format Extraction
To process tables stored as CSV files, run:
```bash
python code/StructuredFormatasInput.py
```

### 3. OCR-Based Extraction
To process tables after applying OCR, run:
```bash
python code/OCRasInput.py
```

## Evaluation
MatSciTableExtract includes multiple evaluation methods to assess extraction accuracy.

### 1. Property Extraction Evaluation

#### Evaluation Scripts:
- **Property name evaluation:**
  ```bash
  python code/F1score-properties_with_missing.py
  ```
- **Detailed property evaluation (including all parameters):**
  ```bash
  python code/F1score-properties_with_missing_including_prop_details.py
  ```
- **Property evaluation without missing samples:**
  ```bash
  python code/F1score-propertieswtmissing.py
  ```

#### Evaluation Process:
- **Stage 1: Property Name Matching**
  - Uses Levenshtein distance (0.6 threshold) to handle variations in property naming.
  - Computes initial F1 scores based on property names.
- **Stage 2: Detailed Property Matching**
  - Evaluates values, units, and conditions.
  - Scores based on exact matches for values and units.
  - Normalized scoring between 0 and 1 for complex condition matching.

### 2. Composition Evaluation

#### Evaluation Scripts:
- **Evaluate composition with missing samples:**
  ```bash
  python code/Accuracy-composition_with_missing.py
  ```
- **Evaluate composition without missing samples:**
  ```bash
  python code/Accuracy-composition_wt_missing.py
  ```

#### Evaluation Features:
- **Flexible string matching:**
  - Sub-string comparison for PST, filler, and matrix names.
  - Case-insensitive comparisons.
  - Partial accuracy scoring for composition fields.
- **Special handling for:**
  - Numeric values and percentages.
  - Control samples (0% composition).
  - Missing or invalid samples.

## Performance Metrics

1. **Composition Accuracy:**
   - Overall accuracy: **0.910** (matrix name, filler name, composition, and PST).

2. **Property Extraction:**
   - **Basic F1 score:** **0.863** (property names only).
   - **Detailed F1 score:** **0.419 - 0.769** (including values, units, conditions).

## Citation

If you use this work, please cite:

```bibtex
@article{circi2024well,
  title={How Well Do Large Language Models Understand Tables in Materials Science?},
  author={Circi, Defne and Khalighinejad, Ghazal and Chen, Anlan and Dhingra, Bhuwan and Brinson, L. Catherine},
  journal={Integrating Materials and Manufacturing Innovation},
  volume={13},
  pages={669--687},
  year={2024},
  publisher={Springer}
}
```
