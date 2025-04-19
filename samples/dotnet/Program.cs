using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

string projectDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", ".."));
string repoDir = Path.GetFullPath(Path.Combine(projectDir, "..", ".."));
string onnxDir = Path.Combine(repoDir, "onnx");

string tokenizerPath = Path.Combine(onnxDir, "bge_m3_tokenizer.onnx");
string modelPath = Path.Combine(onnxDir, "model.onnx");

string text = "A test text! Texto de prueba! Текст для теста! 測試文字! Testtext! Testez le texte! Сынақ мәтіні! Тестни текст! परीक्षण पाठ! Kiểm tra văn bản!";

// Load tokenizer session with Onnx Extensions
SessionOptions tokenizerOptions = new();
tokenizerOptions.RegisterOrtExtensions();
using var tokenizerSession = new InferenceSession(tokenizerPath, tokenizerOptions);

// Create input tensor for tokenizer
// String tensor with shape [1]
var stringTensor = new DenseTensor<string>([1]);
stringTensor[0] = text;

// Create input for tokenizer using CreateFromTensor
var tokenizerInputs = new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor("inputs", stringTensor)
};

// Run tokenizer
using var tokenizerResults = tokenizerSession.Run(tokenizerInputs);
var tokenizerResultsList = tokenizerResults.ToList();

// Extract tokens and token_indices (order: tokens, instance_indices, token_indices)
var tokens = tokenizerResultsList[0].AsTensor<int>().ToArray();
var tokenIndices = tokenizerResultsList[2].AsTensor<int>().ToArray();

Console.WriteLine($"Tokens count: {tokens.Length}");
Console.WriteLine($"Token indices count: {tokenIndices.Length}");

// Convert to input_ids by sorting tokens based on token_indices
var tokenPairs = tokens.Zip(tokenIndices, (t, i) => (token: t, index: i))
    .OrderBy(p => p.index)
    .Select(p => p.token)
    .ToArray();

// Create input_ids tensor with shape [1, tokenPairs.Length]
var inputIdsTensor = new DenseTensor<long>([1, tokenPairs.Length]);
for (int i = 0; i < tokenPairs.Length; i++)
{
    inputIdsTensor[0, i] = tokenPairs[i];
}

// Create attention_mask as all 1s with same shape as input_ids
var attentionMaskTensor = new DenseTensor<long>([1, tokenPairs.Length]);
for (int i = 0; i < tokenPairs.Length; i++)
{
    attentionMaskTensor[0, i] = 1;
}

using var modelSession = new InferenceSession(modelPath);
var modelInputs = new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
    NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor)
};

using var modelResults = modelSession.Run(modelInputs);
var modelResultsList = modelResults.ToList();
var sentenceEmbedding = modelResultsList[1].AsTensor<float>().ToArray();

Console.WriteLine($"Generated embedding length: {sentenceEmbedding.Length}");
Console.WriteLine($"Sample values: {string.Join(", ", sentenceEmbedding.Take(5))}");