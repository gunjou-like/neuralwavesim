<#
.SYNOPSIS
    Check current GCP costs and usage
.PARAMETER ProjectId
    GCP Project ID
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectId
)

Write-Host "💰 Checking costs for project: $ProjectId" -ForegroundColor Cyan
Write-Host ""

# Set project
gcloud config set project $ProjectId --quiet

# Get service list
Write-Host "📋 Cloud Run Services:" -ForegroundColor Yellow
gcloud run services list --region asia-northeast1 --format="table(SERVICE,URL,LAST_DEPLOYED,STATUS)"

Write-Host ""
Write-Host "📊 Recent metrics (last 7 days):" -ForegroundColor Yellow
Write-Host "   (Opening Cloud Console for detailed metrics...)" -ForegroundColor Cyan

# Open billing page
Start-Process "https://console.cloud.google.com/billing?project=$ProjectId"

# Open Cloud Run metrics
Start-Process "https://console.cloud.google.com/run?project=$ProjectId"

Write-Host ""
Write-Host "💡 Tips:" -ForegroundColor Cyan
Write-Host "   - Check 'Request count' and 'Container instance time' metrics" -ForegroundColor White
Write-Host "   - Verify min-instances=0 (no idle costs)" -ForegroundColor White
Write-Host "   - Review container CPU and memory usage" -ForegroundColor White
Write-Host ""
Write-Host "🚨 If costs are high:" -ForegroundColor Red
Write-Host "   1. Run: [emergency_stop.ps1](http://_vscodecontentref_/12) -ProjectId `"$ProjectId`"" -ForegroundColor White
Write-Host "   2. Check for unexpected traffic or errors" -ForegroundColor White
Write-Host "   3. Consider reducing max-instances" -ForegroundColor White