<#
.SYNOPSIS
    Emergency stop all Cloud Run services
.PARAMETER ProjectId
    GCP Project ID
.PARAMETER Region
    GCP Region (default: asia-northeast1 - Tokyo)
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectId,
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "asia-northeast1"  # ✅ 東京リージョン
)

Write-Host "🚨 Emergency Stop - Disabling all Cloud Run services" -ForegroundColor Red
Write-Host "   Project: $ProjectId" -ForegroundColor Yellow
Write-Host "   Region : $Region" -ForegroundColor Yellow

# Set project
gcloud config set project $ProjectId

# List all services
Write-Host ""
Write-Host "📋 Current services:" -ForegroundColor Cyan
gcloud run services list --region $Region --format="table(SERVICE,URL,STATUS)"

Write-Host ""
$confirm = Read-Host "Are you sure you want to stop all services? (yes/no)"
if ($confirm -ne "yes") {
    Write-Host "❌ Cancelled." -ForegroundColor Yellow
    exit 0
}

# Stop API
Write-Host ""
Write-Host "Stopping neuralwavesim-api..." -ForegroundColor Yellow
gcloud run services update neuralwavesim-api `
  --min-instances 0 `
  --max-instances 0 `
  --region $Region `
  --quiet

# Stop UI
Write-Host "Stopping neuralwavesim-ui..." -ForegroundColor Yellow
gcloud run services update neuralwavesim-ui `
  --min-instances 0 `
  --max-instances 0 `
  --region $Region `
  --quiet

Write-Host ""
Write-Host "✅ All services stopped." -ForegroundColor Green
Write-Host "   No new requests will be processed." -ForegroundColor Yellow
Write-Host "   Existing requests will complete." -ForegroundColor Yellow
Write-Host ""
Write-Host "💰 Cost impact:" -ForegroundColor Cyan
Write-Host "   - No new charges will be incurred" -ForegroundColor White
Write-Host "   - Container images remain in GCR (minimal storage cost)" -ForegroundColor White
Write-Host ""
Write-Host "To re-enable services, run:" -ForegroundColor Cyan
Write-Host "   gcloud run services update neuralwavesim-api --max-instances 10 --region $Region" -ForegroundColor White
Write-Host "   gcloud run services update neuralwavesim-ui --max-instances 10 --region $Region" -ForegroundColor White
Write-Host ""
Write-Host "Or simply re-run:" -ForegroundColor Cyan
Write-Host "   .\deploy.ps1 -ProjectId `"$ProjectId`"" -ForegroundColor White