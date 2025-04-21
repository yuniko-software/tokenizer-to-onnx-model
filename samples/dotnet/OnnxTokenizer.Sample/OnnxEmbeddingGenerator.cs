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
        // Create input tensor for tokenizer - np.array([text]) equivalent
        var tokenizerInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("inputs", new DenseTensor<string>(new[] { text }, new[] { 1 }))
        };

        // Run tokenizer
        using var tokenizerOutputs = _tokenizerSession.Run(tokenizerInputs);

        // Extract tokens and token_indices (order: tokens, instance_indices, token_indices)
        // This matches the Python: tokens, _, token_indices = tokenizer_outputs
        var tokens = tokenizerOutputs.ElementAt(0).AsTensor<int>().ToArray();
        var tokenIndices = tokenizerOutputs.ElementAt(2).AsTensor<int>().ToArray();

        // Convert tokenizer outputs to model inputs - similar to convert_tokenizer_outputs in Python
        var tokenPairs = new List<(int index, int token)>();
        for (int i = 0; i < tokens.Length && i < tokenIndices.Length; i++)
        {
            tokenPairs.Add((tokenIndices[i], tokens[i]));
        }

        // Sort by indices
        tokenPairs.Sort();
        var orderedTokens = tokenPairs.Select(p => p.token).ToArray();

        // Create input_ids tensor - equivalent to np.array([ordered_tokens], dtype=np.int64)
        var inputIdsTensor = new DenseTensor<long>(new[] { 1, orderedTokens.Length });
        for (int i = 0; i < orderedTokens.Length; i++)
        {
            inputIdsTensor[0, i] = orderedTokens[i];
        }

        // Create attention_mask - equivalent to np.ones_like(input_ids, dtype=np.int64)
        var attentionMaskTensor = new DenseTensor<long>(new[] { 1, orderedTokens.Length });
        for (int i = 0; i < orderedTokens.Length; i++)
        {
            attentionMaskTensor[0, i] = 1;
        }

        // Run model - equivalent to model_session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
        var modelInputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMaskTensor)
        };

        using var modelOutputs = _modelSession.Run(modelInputs);

        // Extract sentence embedding - like return outputs[1] in Python
        var sentenceEmbedding = modelOutputs.ElementAt(1).AsTensor<float>().ToArray();

        return sentenceEmbedding;
    }

    /// <summary>
    /// Calculates cosine similarity between two embedding vectors.
    /// </summary>
    /// <param name="vectorA">First embedding vector.</param>
    /// <param name="vectorB">Second embedding vector.</param>
    /// <returns>Cosine similarity value between -1 and 1.</returns>
    public static double CalculateCosineSimilarity(float[] vectorA, float[] vectorB)
    {
        if (vectorA.Length != vectorB.Length)
            throw new ArgumentException("Vectors must be of the same length");

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < vectorA.Length; i++)
        {
            dotProduct += vectorA[i] * vectorB[i];
            normA += Math.Pow(vectorA[i], 2);
            normB += Math.Pow(vectorB[i], 2);
        }

        return dotProduct / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }

    /// <summary>
    /// Disposes the resources used by the OnnxEmbeddingGenerator.
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Disposes the resources used by the OnnxEmbeddingGenerator.
    /// </summary>
    /// <param name="disposing">True if disposing managed resources.</param>
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

    /// <summary>
    /// Finalizer.
    /// </summary>
    ~OnnxEmbeddingGenerator()
    {
        Dispose(false);
    }
}