#!/bin/bash
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting ONNX tokenizer model tests${NC}"

# Verify ONNX models exist
if [ ! -d "onnx" ]; then
    echo -e "${RED}ERROR: onnx directory not found!${NC}"
    echo "Please create an 'onnx' directory in the repository root and add tokenizer.onnx and model.onnx files."
    exit 1
fi

if [ ! -f "onnx/tokenizer.onnx" ]; then
    echo -e "${RED}ERROR: tokenizer.onnx not found!${NC}"
    echo "Please download or generate the tokenizer ONNX model and place it in the onnx directory."
    exit 1
fi

if [ ! -f "onnx/model.onnx" ]; then
    echo -e "${RED}ERROR: model.onnx not found!${NC}"
    echo "Please download the model ONNX file and place it in the onnx directory."
    exit 1
fi

# Step 1: Generate reference embeddings using Python
echo -e "${YELLOW}Generating reference embeddings using Python...${NC}"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: python3 command not found!${NC}"
    echo "Please install Python 3 to run this test script."
    exit 1
fi

# Check if required packages are installed
PACKAGES=("onnx==1.16.0" "onnxruntime" "onnxruntime-extensions" "numpy" "transformers")
MISSING_PACKAGES=()

for pkg in "${PACKAGES[@]}"; do
    # Convert dashes to underscores for import check, remove version specification
    pkg_name=${pkg%%==*} # Remove version if present (e.g., onnx==1.16.0 -> onnx)
    pkg_import=${pkg_name//-/_}
    if ! python3 -c "import $pkg_import" &> /dev/null; then
        MISSING_PACKAGES+=("$pkg")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${YELLOW}Installing required Python packages: ${MISSING_PACKAGES[*]}${NC}"
    pip install "${MISSING_PACKAGES[@]}"
fi

# Run the Python script to generate reference embeddings
python3 generate_reference_embeddings.py
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Failed to generate reference embeddings!${NC}"
    exit 1
fi

echo -e "${GREEN}Reference embeddings generated successfully!${NC}"

# Step 2: Run .NET tests
echo -e "${YELLOW}Running .NET tests...${NC}"

cd samples/dotnet
dotnet test --verbosity normal

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: .NET tests failed!${NC}"
    exit 1
fi

echo -e "${GREEN}All tests passed successfully!${NC}"