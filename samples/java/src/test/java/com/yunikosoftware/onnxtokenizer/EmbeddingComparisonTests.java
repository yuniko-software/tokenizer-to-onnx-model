package com.yunikosoftware.onnxtokenizer;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.AfterAll;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

public class EmbeddingComparisonTests {
    private static OnnxEmbeddingGenerator embeddingGenerator;
    private static Map<String, float[]> referenceEmbeddings;

    @BeforeAll
    public static void setup() throws Exception {
        Path tokenizerPath = resolveModelPath("tokenizer.onnx");
        Path modelPath = resolveModelPath("model.onnx");
        Path referenceFile = resolveModelPath("reference_embeddings.json");

        if (!tokenizerPath.toFile().exists())
            throw new FileNotFoundException("Tokenizer file not found at " + tokenizerPath);
        if (!modelPath.toFile().exists())
            throw new FileNotFoundException("Model file not found at " + modelPath);
        if (!referenceFile.toFile().exists())
            throw new FileNotFoundException("Reference embeddings file not found at " + referenceFile);

        embeddingGenerator = new OnnxEmbeddingGenerator(tokenizerPath.toString(), modelPath.toString());

        ObjectMapper mapper = new ObjectMapper();
        referenceEmbeddings = mapper.readValue(
            new FileReader(referenceFile.toFile()),
            new TypeReference<Map<String, float[]>>() {}
        );
    }

    @Test
    public void javaEmbeddings_ShouldMatchPythonEmbeddings() throws Exception {
        for (Map.Entry<String, float[]> entry : referenceEmbeddings.entrySet()) {
            String text = entry.getKey();
            float[] referenceEmbedding = entry.getValue();

            float[] javaEmbedding = embeddingGenerator.generateEmbedding(text);

            double similarity = calculateCosineSimilarity(javaEmbedding, referenceEmbedding);

            assertTrue(similarity > 0.9999,
                "Expected similarity > 0.9999, but got " + similarity + " for text: " + text);
        }
    }

    private static double calculateCosineSimilarity(float[] vectorA, float[] vectorB) {
        if (vectorA.length != vectorB.length)
            throw new IllegalArgumentException("Vectors must be of the same length");

        double dotProduct = 0;
        double normA = 0;
        double normB = 0;

        for (int i = 0; i < vectorA.length; i++) {
            dotProduct += vectorA[i] * vectorB[i];
            normA += vectorA[i] * vectorA[i];
            normB += vectorB[i] * vectorB[i];
        }

        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    private static Path resolveModelPath(String filename) throws FileNotFoundException {
        Path currentDir = Paths.get("").toAbsolutePath();

        for (int i = 0; i < 5; i++) {
            Path onnxDir = currentDir.resolve("onnx");

            if (onnxDir.toFile().exists() && onnxDir.toFile().isDirectory()) {
                return onnxDir.resolve(filename);
            }

            Path parent = currentDir.getParent();
            if (parent == null)
                break;
            currentDir = parent;
        }

        throw new FileNotFoundException("Could not locate '" + filename + "' in the 'onnx' directory");
    }

    @AfterAll
    public static void cleanup() throws Exception {
        if (embeddingGenerator != null) {
            embeddingGenerator.close();
        }
    }
}