﻿using OnnxTokenizer.Sample;

string projectDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", ".."));
string repoDir = Path.GetFullPath(Path.Combine(projectDir, "..", "..", ".."));
string onnxDir = Path.Combine(repoDir, "onnx");

string tokenizerPath = Path.Combine(onnxDir, "tokenizer.onnx");
string modelPath = Path.Combine(onnxDir, "model.onnx");

// Sample text to test with
string text = "A test text! Texto de prueba! Текст для теста! 測試文字! Testtext! Testez le texte! Сынақ мәтіні! Тестни текст! परीक्षण पाठ! Kiểm tra văn bản!";

// Create the embedding generator
using var embeddingGenerator = new OnnxEmbeddingGenerator(tokenizerPath, modelPath);

// Generate embedding
var embedding = embeddingGenerator.GenerateEmbedding(text);
Console.WriteLine($"Generated embedding length: {embedding.Length}");
Console.WriteLine($"Sample values: {string.Join(", ", embedding.Take(5))}");