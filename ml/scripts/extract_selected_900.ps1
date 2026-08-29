$ErrorActionPreference = "Stop"

$DATA_ROOT = "D:\PS26143_DATA"
$RAW = Join-Path $DATA_ROOT "raw"
$MANIFEST_DIR = Join-Path $DATA_ROOT "manifests"
$OUT = Join-Path $DATA_ROOT "selected_900"
$SEVENZIP = "7z"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "PS26143 - SELECTED 900 EXTRACTION" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if (-not (Get-Command $SEVENZIP -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: 7z not found in PATH." -ForegroundColor Red
    exit 1
}

Write-Host "7-Zip: OK" -ForegroundColor Green

$manifestFiles = @(
    "train_manifest.csv",
    "val_manifest.csv",
    "test_manifest.csv"
)

foreach ($mf in $manifestFiles) {
    $path = Join-Path $MANIFEST_DIR $mf

    if (-not (Test-Path $path)) {
        Write-Host "MISSING: $path" -ForegroundColor Red
        exit 1
    }
}

$train = Import-Csv (Join-Path $MANIFEST_DIR "train_manifest.csv")
$val = Import-Csv (Join-Path $MANIFEST_DIR "val_manifest.csv")
$test = Import-Csv (Join-Path $MANIFEST_DIR "test_manifest.csv")

$all = @($train) + @($val) + @($test)

Write-Host ""
Write-Host "TRAIN : $($train.Count)"
Write-Host "VAL   : $($val.Count)"
Write-Host "TEST  : $($test.Count)"
Write-Host "TOTAL : $($all.Count)"

if ($all.Count -ne 900) {
    Write-Host "FATAL: Expected exactly 900 samples." -ForegroundColor Red
    exit 1
}

$duplicates = $all |
    Group-Object global_id |
    Where-Object { $_.Count -gt 1 }

if ($duplicates) {
    Write-Host "FATAL: Duplicate global IDs detected." -ForegroundColor Red
    $duplicates | Format-Table
    exit 1
}

Write-Host "900 unique global IDs confirmed." -ForegroundColor Green

$archives = @{
    oil_image = Join-Path $RAW "zenodo_part1\01_Train_Val_Oil_Spill_images.7z"
    oil_mask = Join-Path $RAW "zenodo_part1\01_Train_Val_Oil_Spill_mask.7z"
    lookalike_image = Join-Path $RAW "zenodo_part2\01_Train_Val_Lookalike_images.7z"
    lookalike_mask = Join-Path $RAW "zenodo_part2\01_Train_Val_Lookalike_mask.7z"
    no_oil_image = Join-Path $RAW "zenodo_part2\01_Train_Val_No_Oil_Images.7z"
    no_oil_mask = Join-Path $RAW "zenodo_part2\01_Train_Val_No_Oil_mask.7z"
}

foreach ($key in $archives.Keys) {
    if (-not (Test-Path $archives[$key])) {
        Write-Host "MISSING ARCHIVE: $($archives[$key])" -ForegroundColor Red
        exit 1
    }
}

function Extract-Selected {
    param(
        [string]$Dataset,
        [string]$Archive,
        [string]$Column,
        [string]$OutputDir
    )

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host "DATASET : $Dataset" -ForegroundColor Yellow
    Write-Host "ARCHIVE : $Archive" -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Yellow

    New-Item -ItemType Directory -Force $OutputDir | Out-Null

    $members = @(
        $all |
        Where-Object { $_.dataset -eq $Dataset } |
        ForEach-Object { $_.$Column } |
        Sort-Object -Unique
    )

    Write-Host "Selected members: $($members.Count)"

    if ($members.Count -eq 0) {
        return
    }

    $listFile = Join-Path $env:TEMP "ps26143_$Dataset`_$([guid]::NewGuid()).txt"

    try {
        $members | Set-Content -Path $listFile -Encoding UTF8

        Write-Host ""
        Write-Host "Starting 7-Zip extraction..."
        Write-Host "Archive is solid; extraction may take time."
        Write-Host ""

        & $SEVENZIP x `
            $Archive `
            "-o$OutputDir" `
            "@$listFile" `
            "-y" `
            "-bb1"

        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "FATAL: 7-Zip failed with exit code $LASTEXITCODE." -ForegroundColor Red
            exit $LASTEXITCODE
        }

        Write-Host ""
        Write-Host "Extraction complete." -ForegroundColor Green
    }
    finally {
        if (Test-Path $listFile) {
            Remove-Item $listFile -Force
        }
    }
}

Extract-Selected `
    -Dataset "oil" `
    -Archive $archives.oil_image `
    -Column "image_member" `
    -OutputDir (Join-Path $OUT "oil\images")

Extract-Selected `
    -Dataset "lookalike" `
    -Archive $archives.lookalike_image `
    -Column "image_member" `
    -OutputDir (Join-Path $OUT "lookalike\images")

Extract-Selected `
    -Dataset "no_oil" `
    -Archive $archives.no_oil_image `
    -Column "image_member" `
    -OutputDir (Join-Path $OUT "no_oil\images")

Extract-Selected `
    -Dataset "oil" `
    -Archive $archives.oil_mask `
    -Column "mask_member" `
    -OutputDir (Join-Path $OUT "oil\masks")

Extract-Selected `
    -Dataset "lookalike" `
    -Archive $archives.lookalike_mask `
    -Column "mask_member" `
    -OutputDir (Join-Path $OUT "lookalike\masks")

Extract-Selected `
    -Dataset "no_oil" `
    -Archive $archives.no_oil_mask `
    -Column "mask_member" `
    -OutputDir (Join-Path $OUT "no_oil\masks")

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "EXTRACTION SUMMARY" -ForegroundColor Cyan
Write-Host "============================================================"

foreach ($dataset in @("oil", "lookalike", "no_oil")) {

    $imageDir = Join-Path $OUT "$dataset\images"
    $maskDir = Join-Path $OUT "$dataset\masks"

    $images = @(Get-ChildItem $imageDir -Recurse -Filter *.tif)
    $masks = @(Get-ChildItem $maskDir -Recurse -Filter *.tif)

    Write-Host ""
    Write-Host $dataset.ToUpper()
    Write-Host "  Images: $($images.Count)"
    Write-Host "  Masks : $($masks.Count)"
}

Write-Host ""
Write-Host "============================================================"
Write-Host "EXTRACTION FINISHED"
Write-Host "============================================================"
Write-Host ""
Write-Host "Output:"
Write-Host $OUT