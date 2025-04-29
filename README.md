# Hugging Face Tokenizer to ONNX Model

This repository demonstrates how to convert Hugging Face tokenizers to ONNX format and use them along with embedding models in multiple programming languages.

## Key Features

- Generate embeddings directly in C#, Java, or Python without third-party APIs or services
- Reduced latency with local embedding generation
- Full control over the embedding pipeline with no external dependencies
- Works offline without internet connectivity requirements
- Cross-platform compatibility

## The Problem

While we can easily download ONNX models from Hugging Face or convert existing PyTorch models to ONNX format for portability, **tokenizers** present a significant challenge.

Tokenizers for embedding models are not often implemented in languages other than Python. This becomes a major obstacle when trying to use embedding models in languages like C# or Java. Developers face the difficult choice of either implementing complex tokenizers from scratch or relying on Python interop, which adds complexity and dependencies.

## The Solution

This repository uses ONNX Runtime Extensions to convert Hugging Face tokenizers to ONNX format. This gives you the complete embedding pipeline in your preferred programming language without having to implement tokenizers yourself.

ONNX Runtime Extensions are currently supported in:
- C#
- Java
- C++
- Python

## Repository Structure

- `tokenizer_to_onnx_model.ipynb` - Jupyter notebook demonstrating the tokenizer conversion process
- `/samples/dotnet` - C# implementation and tests
- `/samples/java` - Java implementation and tests
- `generate_reference_embeddings.py` - Script to generate reference embeddings for cross-language testing
- `run_tests.sh` and `run_tests.ps1` - Test scripts for Linux/macOS and Windows

## Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/yuniko-software/tokenizer-to-onnx-model.git
   cd tokenizer-to-onnx-model
   ```

2. Download the embedding model:
   - Create an `onnx` folder in the repository root
   - Download `model.onnx` and `model.onnx_data` from https://huggingface.co/BAAI/bge-m3/tree/main/onnx
   - Place these files in the `/onnx` folder
   
   > Note: In this repository, we use `bge-m3` as the embedding model and `XLM-RoBERTa Fast` as the tokenizer.

3. Generate the ONNX tokenizer:
   - Option 1: Run the Jupyter notebook
     - Open and run `tokenizer_to_onnx_model.ipynb` - this is the most important file in the repository
     - The notebook demonstrates how to convert a Hugging Face tokenizer to ONNX format
     - This will create a `tokenizer.onnx` file in the `/onnx` folder
   
   - Option 2: Download pre-converted files
     - Check the repository releases and download `onnx.zip`
     - It already contains the bge-m3 embedding model and its tokenizer

4. Run the samples:
   - Once you have `tokenizer.onnx`, `model.onnx`, and `model.onnx_data` in the `/onnx` folder, you can run any sample
   - Try the .NET sample in `/samples/dotnet` or the Java sample in `/samples/java`

5. Verify cross-language embeddings (optional):
   - To ensure that .NET and Java embeddings match the HuggingFace-generated embeddings, you can run:
   
   - On Linux/macOS:
     ```bash
     chmod +x run_tests.sh
     ./run_tests.sh
     ```
   
   - On Windows:
     ```powershell
     ./run_tests.ps1
     ```
   
   > Note: These scripts are primarily used for CI in this repository, but you can run them to verify everything works correctly. They require Python, .NET, Java, and Maven to be installed.

## Python Example

```python
import onnxruntime as ort
import numpy as np
from onnxruntime_extensions import get_library_path

def generate_embedding(text, tokenizer_session, model_session):
    tokenizer_outputs = tokenizer_session.run(None, {"inputs": np.array([text])})
    tokens, _, token_indices = tokenizer_outputs
    
    token_pairs = []
    for i in range(len(tokens)):
        if i < len(token_indices):
            token_pairs.append((token_indices[i], tokens[i]))
    
    token_pairs.sort()
    ordered_tokens = [pair[1] for pair in token_pairs]
    
    input_ids = np.array([ordered_tokens], dtype=np.int64)
    attention_mask = np.ones_like(input_ids, dtype=np.int64)
    
    outputs = model_session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    })
    
    return outputs[1].flatten().tolist()

# Initialize sessions
sess_options = ort.SessionOptions()
sess_options.register_custom_ops_library(get_library_path())
tokenizer_session = ort.InferenceSession("onnx/tokenizer.onnx", sess_options=sess_options)
model_session = ort.InferenceSession("onnx/model.onnx")

# Generate embedding
embedding = generate_embedding("Hello world!", tokenizer_session, model_session)

# See full implementation in tokenizer_to_onnx_model.ipynb
```

## C# Example

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// Initialize sessions
var tokenizerOptions = new SessionOptions();
tokenizerOptions.RegisterOrtExtensions();
var tokenizerSession = new InferenceSession("onnx/tokenizer.onnx", tokenizerOptions);
var modelSession = new InferenceSession("onnx/model.onnx");

// Run tokenizer and model
// See full implementation in samples/dotnet
```

## Java Example

```java
import ai.onnxruntime.*;
import ai.onnxruntime.extensions.OrtxPackage;

// Initialize sessions
OrtSession.SessionOptions tokenizerOptions = new OrtSession.SessionOptions();
tokenizerOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
tokenizerSession = environment.createSession("onnx/tokenizer.onnx", tokenizerOptions);
modelSession = environment.createSession("onnx/model.onnx");

// Run tokenizer and model
// See full implementation in samples/java
```
---

⭐ **If you find this project useful, please consider giving it a star on GitHub!** ⭐ 

Your support helps make this project more visible to other developers who might benefit from it.


