using Newtonsoft.Json;
using OnnxTokenizer.Sample;

namespace OnnxTokenizer.Tests;

public class EmbeddingComparisonTests : IDisposable
{
    private readonly OnnxEmbeddingGenerator _embeddingGenerator;
    private readonly Dictionary<string, float[]> _referenceEmbeddings;

    public EmbeddingComparisonTests()
    {
        string projectDir = Path.GetFullPath(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "..", "..", ".."));
        string repoDir = Path.GetFullPath(Path.Combine(projectDir, "..", "..", ".."));
        string onnxDir = Path.Combine(repoDir, "onnx");
        string tokenizerPath = Path.Combine(onnxDir, "tokenizer.onnx");
        string modelPath = Path.Combine(onnxDir, "model.onnx");
        string referenceFile = Path.Combine(onnxDir, "reference_embeddings.json");

        if (!File.Exists(tokenizerPath))
            throw new FileNotFoundException($"Tokenizer file not found at {tokenizerPath}");
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Model file not found at {modelPath}");
        if (!File.Exists(referenceFile))
            throw new FileNotFoundException($"Reference embeddings file not found at {referenceFile}");

        _embeddingGenerator = new OnnxEmbeddingGenerator(tokenizerPath, modelPath);

        string jsonContent = File.ReadAllText(referenceFile);
        _referenceEmbeddings = JsonConvert.DeserializeObject<Dictionary<string, float[]>>(jsonContent)!;
    }

    [Fact]
    public void NetEmbeddings_ShouldMatchPythonEmbeddings()
    {
        foreach (var entry in _referenceEmbeddings)
        {
            string text = entry.Key;
            float[] referenceEmbedding = entry.Value;

            // Generate embedding using .NET implementation
            float[] dotNetEmbedding = _embeddingGenerator.GenerateEmbedding(text);

            // Calculate cosine similarity
            double similarity = CalculateCosineSimilarity(dotNetEmbedding, referenceEmbedding);

            // Assert high similarity (practically identical)
            Assert.True(similarity > 0.9999,
                $"Expected similarity > 0.9999, but got {similarity} for text: {text}");
        }
    }

    private static double CalculateCosineSimilarity(float[] vectorA, float[] vectorB)
    {
        if (vectorA.Length != vectorB.Length)
            throw new ArgumentException("Vectors must be of the same length");

        double dotProduct = 0;
        double normA = 0;
        double normB = 0;

        for (int i = 0; i < vectorA.Length; i++)
        {
            dotProduct += vectorA[i] * vectorB[i];
            normA += vectorA[i] * vectorA[i];
            normB += vectorB[i] * vectorB[i];
        }

        return dotProduct / (Math.Sqrt(normA) * Math.Sqrt(normB));
    }

    public void Dispose()
    {
        _embeddingGenerator?.Dispose();
    }
}