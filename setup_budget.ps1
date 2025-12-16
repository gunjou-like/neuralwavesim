<#
.SYNOPSIS
    Setup GCP Budget Alerts with optional auto-stop
.PARAMETER ProjectId
    GCP Project ID
.PARAMETER BudgetAmount
    Monthly budget in USD (default: 10)
.PARAMETER EnableAutoStop
    Enable automatic service stop when budget is exceeded (default: false)
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectId,
    
    [Parameter(Mandatory=$false)]
    [int]$BudgetAmount = 10,
    
    [Parameter(Mandatory=$false)]
    [switch]$EnableAutoStop = $false
)

Write-Host "💰 Setting up budget alerts for project: $ProjectId" -ForegroundColor Cyan
Write-Host "   Budget limit: `$$BudgetAmount USD/month" -ForegroundColor Yellow
Write-Host "   Auto-stop   : $EnableAutoStop" -ForegroundColor Yellow

# Get billing account ID
$BillingAccountId = gcloud billing projects describe $ProjectId --format='value(billingAccountName)' | Split-Path -Leaf

if (-not $BillingAccountId) {
    Write-Host "❌ No billing account found. Please link a billing account first." -ForegroundColor Red
    Write-Host "   https://console.cloud.google.com/billing/linkedaccount?project=$ProjectId" -ForegroundColor Yellow
    exit 1
}

Write-Host "   Billing Account: $BillingAccountId" -ForegroundColor Yellow

# Create Pub/Sub topic for budget alerts (if auto-stop enabled)
if ($EnableAutoStop) {
    Write-Host ""
    Write-Host "📡 Creating Pub/Sub topic for budget alerts..." -ForegroundColor Yellow
    
    gcloud pubsub topics create budget-alerts --project $ProjectId 2>$null
    
    Write-Host "⚠️  Auto-stop feature requires Cloud Functions setup." -ForegroundColor Yellow
    Write-Host "    Manual setup required:" -ForegroundColor Cyan
    Write-Host "    1. Create Cloud Function to listen to 'budget-alerts' topic" -ForegroundColor White
    Write-Host "    2. Function should call [emergency_stop.ps1](http://_vscodecontentref_/10) when triggered" -ForegroundColor White
    Write-Host "    3. See: https://cloud.google.com/billing/docs/how-to/budgets-programmatic-notifications" -ForegroundColor White
}

# Create budget
Write-Host ""
Write-Host "📝 Creating budget..." -ForegroundColor Yellow

gcloud billing budgets create `
  --billing-account=$BillingAccountId `
  --display-name="NeuralWaveSim Monthly Budget" `
  --budget-amount=$BudgetAmount `
  --threshold-rule=percent=0.5 `
  --threshold-rule=percent=0.75 `
  --threshold-rule=percent=0.9 `
  --threshold-rule=percent=1.0

Write-Host ""
Write-Host "✅ Budget alerts configured!" -ForegroundColor Green
Write-Host "   You will receive email notifications at:" -ForegroundColor Cyan
Write-Host "   - 50% of budget (`$$($BudgetAmount * 0.5) USD)" -ForegroundColor White
Write-Host "   - 75% of budget (`$$($BudgetAmount * 0.75) USD)" -ForegroundColor White
Write-Host "   - 90% of budget (`$$($BudgetAmount * 0.9) USD)" -ForegroundColor White
Write-Host "   - 100% of budget (`$$BudgetAmount USD)" -ForegroundColor White
Write-Host ""
Write-Host "📊 View budget status:" -ForegroundColor Cyan
Write-Host "   https://console.cloud.google.com/billing/budgets?project=$ProjectId" -ForegroundColor White
Write-Host ""
Write-Host "💡 Tips:" -ForegroundColor Cyan
Write-Host "   - Monitor costs daily: https://console.cloud.google.com/billing?project=$ProjectId" -ForegroundColor White
Write-Host "   - Manual emergency stop: [emergency_stop.ps1](http://_vscodecontentref_/11) -ProjectId `"$ProjectId`"" -ForegroundColor White
Write-Host "   - View service metrics: https://console.cloud.google.com/run?project=$ProjectId" -ForegroundColor White