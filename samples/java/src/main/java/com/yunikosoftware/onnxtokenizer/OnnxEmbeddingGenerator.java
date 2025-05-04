package com.yunikosoftware.onnxtokenizer;

import ai.onnxruntime.*;
import ai.onnxruntime.extensions.OrtxPackage;
import java.util.*;

public class OnnxEmbeddingGenerator implements AutoCloseable {
    private final OrtSession tokenizerSession;
    private final OrtSession modelSession;

    /**
     * Initializes a new instance of the OnnxEmbeddingGenerator class.
     *
     * @param tokenizerPath Path to the ONNX tokenizer model.
     * @param modelPath Path to the ONNX embedding model.
     * @throws OrtException If there is an error initializing the ONNX sessions.
     */
    public OnnxEmbeddingGenerator(String tokenizerPath, String modelPath) throws OrtException {
        // Initialize tokenizer session with ONNX Extensions
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        
        // Register the ONNX Runtime Extensions library
        OrtSession.SessionOptions tokenizerOptions = new OrtSession.SessionOptions();
        tokenizerOptions.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
        
        tokenizerSession = environment.createSession(tokenizerPath, tokenizerOptions);
        
        modelSession = environment.createSession(modelPath);
    }

    /**
     * Generates embedding for the input text.
     *
     * @param text The input text.
     * @return The embedding vector as a float array.
     * @throws OrtException If there is an error during inference.
     */
    public float[] generateEmbedding(String text) throws OrtException {
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        
        // Create input tensor for tokenizer
        Map<String, OnnxTensor> tokenizerInputs = new HashMap<>();
        String[] inputArray = new String[]{text};
        
        try (OnnxTensor inputTensor = OnnxTensor.createTensor(env, inputArray)) {
            tokenizerInputs.put("inputs", inputTensor);
            
            // Run tokenizer
            try (OrtSession.Result tokenizerResults = tokenizerSession.run(tokenizerInputs)) {
                // Extract tokens and token_indices (order: tokens, instance_indices, token_indices)
                int[] tokens = ((int[]) tokenizerResults.get(0).getValue());
                int[] tokenIndices = ((int[]) tokenizerResults.get(2).getValue());
                
                // Convert to input_ids by sorting tokens based on token_indices
                List<TokenIndexPair> tokenPairs = new ArrayList<>();
                for (int i = 0; i < tokens.length; i++) {
                    if (i < tokenIndices.length) {
                        tokenPairs.add(new TokenIndexPair(tokens[i], tokenIndices[i]));
                    }
                }
                
                // Sort by index
                tokenPairs.sort(Comparator.comparing(pair -> pair.index));
                
                // Extract sorted tokens
                long[] orderedTokens = tokenPairs.stream()
                        .mapToLong(pair -> pair.token)
                        .toArray();
                
                // Create input_ids tensor with shape [1, orderedTokens.length]
                long[][] inputIds = new long[1][orderedTokens.length];
                for (int i = 0; i < orderedTokens.length; i++) {
                    inputIds[0][i] = orderedTokens[i];
                }
                
                // Create attention_mask as all 1s with same shape as input_ids
                long[][] attentionMask = new long[1][orderedTokens.length];
                for (int i = 0; i < orderedTokens.length; i++) {
                    attentionMask[0][i] = 1;
                }
                
                // Run the model with the prepared inputs
                Map<String, OnnxTensor> modelInputs = new HashMap<>();
                try (OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, inputIds);
                     OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, attentionMask)) {
                    
                    modelInputs.put("input_ids", inputIdsTensor);
                    modelInputs.put("attention_mask", attentionMaskTensor);
                    
                    try (OrtSession.Result modelResults = modelSession.run(modelInputs)) {
                        // Extract the sentence embedding (second output)
                        float[][] sentenceEmbedding = (float[][]) modelResults.get(1).getValue();
                        
                        // Return the flattened embedding vector
                        return sentenceEmbedding[0];
                    }
                }
            }
        }
    }
    
    private static class TokenIndexPair {
        public final long token;
        public final long index;
        
        public TokenIndexPair(long token, long index) {
            this.token = token;
            this.index = index;
        }
    }

    @Override
    public void close() throws Exception {
        if (tokenizerSession != null) {
            tokenizerSession.close();
        }
        if (modelSession != null) {
            modelSession.close();
        }
    }
}