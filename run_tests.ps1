function Write-Green {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Green
}

function Write-Red {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Red
}

function Write-Yellow {
    param([string]$Text)
    Write-Host $Text -ForegroundColor Yellow
}

Write-Yellow "Starting ONNX tokenizer model tests"

# Verify ONNX models exist
if (-not (Test-Path "onnx")) {
    Write-Red "ERROR: onnx directory not found!"
    Write-Host "Please create an 'onnx' directory in the repository root and add tokenizer.onnx and model.onnx files."
    exit 1
}

if (-not (Test-Path "onnx/tokenizer.onnx")) {
    Write-Red "ERROR: tokenizer.onnx not found!"
    Write-Host "Please download or generate the tokenizer ONNX model and place it in the onnx directory."
    exit 1
}

if (-not (Test-Path "onnx/model.onnx")) {
    Write-Red "ERROR: model.onnx not found!"
    Write-Host "Please download the model ONNX file and place it in the onnx directory."
    exit 1
}

# Step 1: Generate reference embeddings using Python
Write-Yellow "Generating reference embeddings using Python..."

# Check if Python is available
try {
    $pythonVersion = python --version
} catch {
    Write-Red "ERROR: Python command not found!"
    Write-Host "Please install Python to run this test script."
    exit 1
}

# Check if required packages are installed
$packages = @("onnxruntime", "onnxruntime_extensions", "numpy")
$missingPackages = @()

foreach ($pkg in $packages) {
    $importCheck = python -c "try:
    import $($pkg.Replace('-', '_'))
    print('OK')
except ImportError:
    print('Missing')"
    
    if ($importCheck -match "Missing") {
        $pipPkg = $pkg -replace "_", "-"
        $missingPackages += $pipPkg
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Yellow "Installing required Python packages: $($missingPackages -join ', ')"
    foreach ($pkg in $missingPackages) {
        pip install $pkg
    }
}

# Run the Python script to generate reference embeddings
try {
    python generate_reference_embeddings.py
    if ($LASTEXITCODE -ne 0) {
        Write-Red "ERROR: Failed to generate reference embeddings!"
        exit 1
    }
} catch {
    Write-Red "ERROR: Failed to generate reference embeddings!"
    Write-Host $_.Exception.Message
    exit 1
}

Write-Green "Reference embeddings generated successfully!"

# Step 2: Run .NET tests
Write-Yellow "Running .NET tests..."

Push-Location "samples/dotnet"
try {
    dotnet test --verbosity normal
    if ($LASTEXITCODE -ne 0) {
        Write-Red "ERROR: .NET tests failed!"
        exit 1
    }
} catch {
    Write-Red "ERROR: .NET tests failed!"
    Write-Host $_.Exception.Message
    exit 1
} finally {
    Pop-Location
}

Write-Green "All tests passed successfully!"