<#
.SYNOPSIS
    NeuralWaveSim Deployment Script for GCP Cloud Run
.PARAMETER ProjectId
    GCP Project ID
.PARAMETER Region
    GCP Region (default: asia-northeast1 - Tokyo)
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectId,
    
    [Parameter(Mandatory=$false)]
    [string]$Region = "asia-northeast1"
)

$ErrorActionPreference = "Stop"

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "🚀 NeuralWaveSim - GCP Cloud Run Deployment" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Project ID: $ProjectId" -ForegroundColor Yellow
Write-Host "Region    : $Region (Tokyo)" -ForegroundColor Yellow
Write-Host ""

# Set project
Write-Host "📌 Setting GCP project..." -ForegroundColor Cyan
gcloud config set project $ProjectId

# ✅ Configure Docker authentication for GCR
Write-Host "🔧 Configuring Docker authentication..." -ForegroundColor Cyan
gcloud auth configure-docker --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "⚠️  Docker authentication failed. Trying alternative method..." -ForegroundColor Yellow
    gcloud auth configure-docker gcr.io --quiet
}

# Verify authentication
Write-Host "✅ Docker authentication configured" -ForegroundColor Green
Write-Host ""

# Build and push API
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "📦 Building API container..." -ForegroundColor Yellow
Write-Host "=" * 70 -ForegroundColor Cyan

docker build -t gcr.io/$ProjectId/neuralwavesim-api:latest -f Dockerfile .
if ($LASTEXITCODE -ne 0) { throw "API build failed" }

Write-Host "📤 Pushing API container to GCR..." -ForegroundColor Yellow
docker push gcr.io/$ProjectId/neuralwavesim-api:latest
if ($LASTEXITCODE -ne 0) { throw "API push failed. Please run: gcloud auth configure-docker" }

# Deploy API
Write-Host ""
Write-Host "🚀 Deploying API to Cloud Run (Tokyo)..." -ForegroundColor Yellow
gcloud run deploy neuralwavesim-api `
  --image gcr.io/$ProjectId/neuralwavesim-api:latest `
  --platform managed `
  --region $Region `
  --allow-unauthenticated `
  --port 8080 `
  --memory 2Gi `
  --cpu 2 `
  --timeout 300 `
  --min-instances 0 `
  --max-instances 10 `
  --cpu-throttling `
  --quiet

if ($LASTEXITCODE -ne 0) { throw "API deployment failed" }

# Get API URL
$ApiUrl = gcloud run services describe neuralwavesim-api `
  --platform managed `
  --region $Region `
  --format 'value(status.url)'

Write-Host ""
Write-Host "✅ API deployed successfully!" -ForegroundColor Green
Write-Host "   URL: $ApiUrl" -ForegroundColor Cyan

# Build and push UI
Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "📦 Building UI container..." -ForegroundColor Yellow
Write-Host "=" * 70 -ForegroundColor Cyan

docker build -t gcr.io/$ProjectId/neuralwavesim-ui:latest -f Dockerfile.streamlit .
if ($LASTEXITCODE -ne 0) { throw "UI build failed" }

Write-Host "📤 Pushing UI container to GCR..." -ForegroundColor Yellow
docker push gcr.io/$ProjectId/neuralwavesim-ui:latest
if ($LASTEXITCODE -ne 0) { throw "UI push failed" }

# Deploy UI
Write-Host ""
Write-Host "🚀 Deploying UI to Cloud Run (Tokyo)..." -ForegroundColor Yellow
gcloud run deploy neuralwavesim-ui `
  --image gcr.io/$ProjectId/neuralwavesim-ui:latest `
  --platform managed `
  --region $Region `
  --allow-unauthenticated `
  --port 8501 `
  --memory 1Gi `
  --cpu 1 `
  --timeout 300 `
  --min-instances 0 `
  --max-instances 10 `
  --cpu-throttling `
  --set-env-vars "API_URL=$ApiUrl" `
  --quiet

if ($LASTEXITCODE -ne 0) { throw "UI deployment failed" }

# Get UI URL
$UiUrl = gcloud run services describe neuralwavesim-ui `
  --platform managed `
  --region $Region `
  --format 'value(status.url)'

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Green
Write-Host "✅ Deployment Complete!" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Access URLs:" -ForegroundColor Cyan
Write-Host "   API: $ApiUrl" -ForegroundColor White
Write-Host "   UI : $UiUrl" -ForegroundColor White
Write-Host ""
Write-Host "📊 View in Cloud Console:" -ForegroundColor Cyan
Write-Host "   https://console.cloud.google.com/run?project=$ProjectId" -ForegroundColor White
Write-Host ""
Write-Host "💰 Cost Estimate (Tokyo region):" -ForegroundColor Cyan
Write-Host "   Free tier: 2M requests/month, 180,000 vCPU-seconds/month" -ForegroundColor White
Write-Host "   Estimated cost: `$0-10/month (typical usage)" -ForegroundColor White
Write-Host ""

# Open UI in browser
$response = Read-Host "Open UI in browser? (Y/n)"
if ($response -ne "n") {
    Write-Host "🌐 Opening browser..." -ForegroundColor Yellow
    Start-Process $UiUrl
}

Write-Host ""
Write-Host "✅ Done!" -ForegroundColor Green
Write-Host ""
Write-Host "📝 Next steps:" -ForegroundColor Cyan
Write-Host "   - Test all models in the UI" -ForegroundColor White
Write-Host "   - Setup budget alerts: .\setup_budget.ps1 -ProjectId `"$ProjectId`" -BudgetAmount 10" -ForegroundColor White
Write-Host "   - Monitor costs: .\check_costs.ps1 -ProjectId `"$ProjectId`"" -ForegroundColor White