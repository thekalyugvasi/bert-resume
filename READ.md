# Resume Classifier

This project fine-tunes a pre-trained language model (`bert-base-uncased`) to classify resumes into 22 predefined job categories. The dataset consists of 2400+ resumes, available in string and PDF formats, and is used to train the model for multi-class classification.

---

## Features

- **Base Model:** `bert-base-uncased`  
- **Task:** Multi-class classification  
- **Job Categories:**
  - `ACCOUNTANT`
  - `ADVOCATE`
  - `AGRICULTURE`
  - `APPAREL`
  - `ARTS`
  - `AUTOMOBILE`
  - `AVIATION`
  - `BANKING`
  - `BPO`
  - `BUSINESS-DEVELOPMENT`
  - `CHEF`
  - `CONSTRUCTION`
  - `CONSULTANT`
  - `DESIGNER`
  - `DIGITAL-MEDIA`
  - `ENGINEERING`
  - `FINANCE`
  - `FITNESS`
  - `HEALTHCARE`
  - `HR`
  - `INFORMATION-TECHNOLOGY`
  - `PUBLIC-RELATIONS`
  - `SALES`
  - `TEACHER`

---

## Dataset

- **Description:** 2400+ resumes organized into 22 job categories.
- **Formats:** Text and PDF files.
- **Preprocessing:**
  - Removal of HTML tags, URLs, punctuation, Unicode, escape sequences, stop words, and extra whitespace.

---

## Model Configuration

- **Base Model:** `bert-base-uncased`
- **Fine-tuning:** Performed for multi-class classification on preprocessed text data.
- **Model Output:** Raw logits for each class.

---

## Model Output and Activation

The model returns raw output logits that can be normalized using the sigmoid activation function:

```python
import torch.nn as nn

sigmoid = nn.Sigmoid()
probs = sigmoid(logits)
```
Probs Variable: Contains predicted probabilities for each class, with values ranging from 0 to 1.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/resume-classifier.git
   cd resume-classifier

2. Install dependencies:
    ```bash
        pip install -r requirements.txt
    ```
## Results

    * Performance Metrics: Accuracy, Precision, Recall, and F1-score.
    * Output: Predicted probabilities for each job category.


