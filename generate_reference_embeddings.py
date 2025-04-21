import json
import onnxruntime as ort
import numpy as np
from onnxruntime_extensions import get_library_path
import os

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

def main():
    repo_root = os.getcwd()
    
    onnx_dir = os.path.join(repo_root, "onnx")
    
    tokenizer_path = os.path.join(onnx_dir, "tokenizer.onnx")
    model_path = os.path.join(onnx_dir, "model.onnx")
    
    output_path = os.path.join(onnx_dir, "reference_embeddings.json")
    
    print(f"Using tokenizer: {tokenizer_path}")
    print(f"Using model: {model_path}")
    print(f"Output will be saved to: {output_path}")

    if not os.path.exists(tokenizer_path):
        print(f"ERROR: Tokenizer file not found at {tokenizer_path}")
        return
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        return

    sess_options = ort.SessionOptions()
    sess_options.register_custom_ops_library(get_library_path())
    tokenizer_session = ort.InferenceSession(
        tokenizer_path, 
        sess_options=sess_options, 
        providers=['CPUExecutionProvider']
    )
    
    model_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    
    test_texts = [
        "This is a simple test text.",
        "Hello world!",
        "A test text! Texto de prueba! Текст для теста! 測試文字! Testtext!",
        "",
        "This is a longer text that should generate a meaningful embedding vector. The embedding model should capture the semantic meaning of this text.",
        "ONNX Runtime is a performance-focused engine for ONNX models.",
        "Text with numbers: 12345 and symbols: !@#$%^&*()",
        "English, Español, Русский, 中文, العربية, हिन्दी"
    ]
    
    embeddings = {}
    for text in test_texts:
        print(f"Generating embedding for: {text[:50]}...")
        embedding = generate_embedding(text, tokenizer_session, model_session)
        embeddings[text] = embedding
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(embeddings)} reference embeddings to {output_path}")

if __name__ == "__main__":
    main()