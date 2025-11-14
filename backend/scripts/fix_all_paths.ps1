# Fix all Python scripts to use correct paths
Write-Host "Fixing all script paths..." -ForegroundColor Yellow

$files = @(
    "validate_with_public_data.py",
    "analyze_both_kaggle_datasets.py",
    "analyze_kaggle_datasets.py",
    "test_parser.py",
    "extract_kaggle_datasets.py",
    "download_all_datasets.py",
    "generate_sample_data.py",
    "create_drone_configs.py"
)

foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "Fixing: $file" -ForegroundColor Cyan
        
        # Read content
        $content = Get-Content $file -Raw
        
        # Replace parent.parent with parent.parent.parent for data and config paths
        $content = $content -replace "Path\(__file__\)\.parent\.parent / 'data'", "Path(__file__).parent.parent.parent / 'data'"
        $content = $content -replace "Path\(__file__\)\.parent\.parent / 'config'", "Path(__file__).parent.parent.parent / 'config'"
        
        # Handle variations with quotes
        $content = $content -replace 'Path\(__file__\)\.parent\.parent / "data"', 'Path(__file__).parent.parent.parent / "data"'
        $content = $content -replace 'Path\(__file__\)\.parent\.parent / "config"', 'Path(__file__).parent.parent.parent / "config"'
        
        # Save back
        Set-Content $file -Value $content
        
        Write-Host "  ✓ Fixed" -ForegroundColor Green
    } else {
        Write-Host "  ⊘ Not found: $file" -ForegroundColor Gray
    }
}

Write-Host "`n✅ All paths fixed!" -ForegroundColor Green
Write-Host "Now all scripts will use project root paths" -ForegroundColor Yellow