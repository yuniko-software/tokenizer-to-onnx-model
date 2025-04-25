package com.yunikosoftware.onnxtokenizer;

import java.io.FileNotFoundException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class App {
    public static void main(String[] args) {
        try {
            Path tokenizerPath = resolveModelPath("tokenizer.onnx");
            Path modelPath = resolveModelPath("model.onnx");

            // Sample text to test with
            String text = "A test text! Texto de prueba! Текст для теста! 測試文字! Testtext! Testez le texte! Сынақ мәтіні! Тестни текст! परीक्षण पाठ! Kiểm tra văn bản!";

            System.out.println("Using tokenizer path: " + tokenizerPath);
            System.out.println("Using model path: " + modelPath);
            System.out.println("Generating embedding for text: " + text);

            // Create the embedding generator and generate embedding
            try (OnnxEmbeddingGenerator embeddingGenerator = new OnnxEmbeddingGenerator(tokenizerPath.toString(),
                    modelPath.toString())) {
                float[] embedding = embeddingGenerator.generateEmbedding(text);

                System.out.println("Generated embedding length: " + embedding.length);
                System.out.println("Sample values: " + formatSampleValues(embedding, 5));
            }
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
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

    private static String formatSampleValues(float[] values, int count) {
        if (values.length == 0)
            return "[]";

        StringBuilder sb = new StringBuilder("[");
        int limit = Math.min(count, values.length);

        for (int i = 0; i < limit; i++) {
            if (i > 0)
                sb.append(", ");
            sb.append(String.format("%.6f", values[i]));
        }

        sb.append("]");
        return sb.toString();
    }
}