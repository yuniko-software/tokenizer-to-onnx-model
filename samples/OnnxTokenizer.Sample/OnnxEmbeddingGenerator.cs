using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxTokenizer.Sample;

/// <summary>
/// Provides functionality to generate embeddings using ONNX tokenizer and embedding models.
/// </summary>
public class OnnxEmbeddingGenerator : IDisposable
{
    private readonly InferenceSession _tokenizerSession;
    private readonly InferenceSession _modelSession;
    private bool _disposed = false;

    /// <summary>
    /// Initializes a new instance of the OnnxEmbeddingGenerator class.
    /// </summary>
    /// <param name="tokenizerPath">Path to the ONNX tokenizer model.</param>
    /// <param name="modelPath">Path to the ONNX embedding model.</param>
    public OnnxEmbeddingGenerator(string tokenizerPath, string modelPath)
    {
        // Initialize tokenizer session with ONNX Extensions
        var tokenizerOptions = new SessionOptions();
        tokenizerOptions.RegisterOrtExtensions();
        _tokenizerSession = new InferenceSession(tokenizerPath, tokenizerOptions);

        // Initialize model session
        _modelSession = new InferenceSession(modelPath);
    }

    /// <summary>
    /// Generates embedding for the input text.
    /// </summary>
    /// <param name="text">The input text.</param>
    /// <returns>The embedding vector as a float array.</returns>
    public float[] GenerateEmbedding(string text)
    {
        // Create input tensor for tokenizer
        var stringTensor = new DenseTensor<string>([1]);
        stringTensor[0] = text;

        // Create input for tokenizer using CreateFromTensor
        var tokenizerInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("inputs", stringTensor)
        };

        // Run tokenizer
        using var tokenizerResults = _tokenizerSession.Run(tokenizerInputs);
        var tokenizerResultsList = tokenizerResults.ToList();

        // Extract tokens and token_indices (order: tokens, instance_indices, token_indices)
        var tokens = tokenizerResultsList[0].AsTensor<int>().ToArray();
        var tokenIndices = tokenizerResultsList[2].AsTensor<int>().ToArray();

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

        // Run the model with the prepared inputs
        var modelInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor)
        };

        using var modelResults = _modelSession.Run(modelInputs);
        var modelResultsList = modelResults.ToList();

        // Extract the sentence embedding
        var sentenceEmbedding = modelResultsList[1].AsTensor<float>().ToArray();

        return sentenceEmbedding;
    }

    /// <summary>
    /// Disposes the resources used by the OnnxEmbeddingGenerator.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _tokenizerSession?.Dispose();
                _modelSession?.Dispose();
            }

            _disposed = true;
        }
    }

    ~OnnxEmbeddingGenerator()
    {
        Dispose(false);
    }
}