$ErrorActionPreference = "Stop"

# ============================================================
# PS26143 — SELECTED 900 EXTRACTION AUDIT
# ============================================================

$ROOT = "D:\PS26143_DATA"
$SELECTED = Join-Path $ROOT "selected_900"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "PS26143 - SELECTED 900 EXTRACTION AUDIT" -ForegroundColor Cyan
Write-Host "============================================================"
Write-Host ""

if (-not (Test-Path $SELECTED)) {
    Write-Host "FATAL: selected_900 directory does not exist:" -ForegroundColor Red
    Write-Host $SELECTED
    exit 1
}

# Actual extraction structure:
#
# selected_900/
#   oil/
#       images/Oil/*.tif
#       masks/Mask_oil/*.tif
#
#   lookalike/
#       images/Lookalike/*.tif
#       masks/Mask_lookalike/*.tif
#
#   no_oil/
#       images/No_oil/*.tif
#       masks/Mask_no_oil/*.tif

$expected = @{
    oil = @{
        images = 420
        masks  = 420
        imageDir = Join-Path $SELECTED "oil\images\Oil"
        maskDir  = Join-Path $SELECTED "oil\masks\Mask_oil"
    }

    lookalike = @{
        images = 240
        masks  = 240
        imageDir = Join-Path $SELECTED "lookalike\images\Lookalike"
        maskDir  = Join-Path $SELECTED "lookalike\masks\Mask_lookalike"
    }

    no_oil = @{
        images = 240
        masks  = 240
        imageDir = Join-Path $SELECTED "no_oil\images\No_oil"
        maskDir  = Join-Path $SELECTED "no_oil\masks\Mask_no_oil"
    }
}

$totalImages = 0
$totalMasks = 0
$allGood = $true

foreach ($dataset in @("oil", "lookalike", "no_oil")) {

    Write-Host ""
    Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
    Write-Host $dataset.ToUpper() -ForegroundColor Yellow
    Write-Host "------------------------------------------------------------"

    $cfg = $expected[$dataset]

    # --------------------------------------------------------
    # Directory checks
    # --------------------------------------------------------

    if (-not (Test-Path $cfg.imageDir)) {
        Write-Host ""
        Write-Host "ERROR: Image directory missing:" -ForegroundColor Red
        Write-Host $cfg.imageDir
        $allGood = $false
        continue
    }

    if (-not (Test-Path $cfg.maskDir)) {
        Write-Host ""
        Write-Host "ERROR: Mask directory missing:" -ForegroundColor Red
        Write-Host $cfg.maskDir
        $allGood = $false
        continue
    }

    # --------------------------------------------------------
    # Count files
    # --------------------------------------------------------

    $images = @(
        Get-ChildItem -Path $cfg.imageDir -Filter "*.tif" -File
    )

    $masks = @(
        Get-ChildItem -Path $cfg.maskDir -Filter "*.tif" -File
    )

    $imageCount = $images.Count
    $maskCount = $masks.Count

    $totalImages += $imageCount
    $totalMasks += $maskCount

    Write-Host ""
    Write-Host "Image directory:"
    Write-Host "  $($cfg.imageDir)"

    Write-Host "Mask directory:"
    Write-Host "  $($cfg.maskDir)"

    Write-Host ""
    Write-Host "Images found :" $imageCount
    Write-Host "Images expected:" $cfg.images

    Write-Host "Masks found  :" $maskCount
    Write-Host "Masks expected:" $cfg.masks

    # --------------------------------------------------------
    # Count validation
    # --------------------------------------------------------

    if ($imageCount -eq $cfg.images) {
        Write-Host "IMAGE COUNT PASS" -ForegroundColor Green
    }
    else {
        Write-Host "IMAGE COUNT FAILED" -ForegroundColor Red
        $allGood = $false
    }

    if ($maskCount -eq $cfg.masks) {
        Write-Host "MASK COUNT PASS" -ForegroundColor Green
    }
    else {
        Write-Host "MASK COUNT FAILED" -ForegroundColor Red
        $allGood = $false
    }

    # --------------------------------------------------------
    # Image / mask ID pairing
    # --------------------------------------------------------

    $imageIDs = @(
        $images |
        ForEach-Object {
            [int]([System.IO.Path]::GetFileNameWithoutExtension($_.Name))
        }
    )

    $maskIDs = @(
        $masks |
        ForEach-Object {
            [int]([System.IO.Path]::GetFileNameWithoutExtension($_.Name))
        }
    )

    $missingMasks = @(
        $imageIDs |
        Where-Object {
            $_ -notin $maskIDs
        } |
        Sort-Object -Unique
    )

    $orphanMasks = @(
        $maskIDs |
        Where-Object {
            $_ -notin $imageIDs
        } |
        Sort-Object -Unique
    )

    Write-Host ""
    Write-Host "Missing masks :" $missingMasks.Count
    Write-Host "Orphan masks  :" $orphanMasks.Count

    if ($missingMasks.Count -gt 0) {
        Write-Host ""
        Write-Host "MISSING MASK IDs:" -ForegroundColor Red

        foreach ($id in $missingMasks) {
            Write-Host ("  {0:D5}" -f $id)
        }

        $allGood = $false
    }

    if ($orphanMasks.Count -gt 0) {
        Write-Host ""
        Write-Host "ORPHAN MASK IDs:" -ForegroundColor Red

        foreach ($id in $orphanMasks) {
            Write-Host ("  {0:D5}" -f $id)
        }

        $allGood = $false
    }

    if (
        $missingMasks.Count -eq 0 -and
        $orphanMasks.Count -eq 0
    ) {
        Write-Host "IMAGE/MASK PAIRING PASS" -ForegroundColor Green
    }
}

# ============================================================
# GLOBAL SUMMARY
# ============================================================

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "GLOBAL SUMMARY" -ForegroundColor Cyan
Write-Host "============================================================"
Write-Host ""

Write-Host "Total images :" $totalImages
Write-Host "Total masks  :" $totalMasks
Write-Host "Expected     : 900 images + 900 masks"

Write-Host ""

if ($totalImages -ne 900) {
    Write-Host "FATAL: Total image count is not 900." -ForegroundColor Red
    $allGood = $false
}

if ($totalMasks -ne 900) {
    Write-Host "FATAL: Total mask count is not 900." -ForegroundColor Red
    $allGood = $false
}

Write-Host ""

if ($allGood) {

    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "AUDIT PASSED" -ForegroundColor Green
    Write-Host "900 IMAGE/MASK PAIRS VERIFIED" -ForegroundColor Green
    Write-Host "============================================================"

}
else {

    Write-Host "============================================================" -ForegroundColor Red
    Write-Host "AUDIT FAILED" -ForegroundColor Red
    Write-Host "DO NOT UPLOAD YET" -ForegroundColor Red
    Write-Host "============================================================"

    exit 1
}