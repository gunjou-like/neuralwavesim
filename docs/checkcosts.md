# 予防
```
# 予算アラートを設定
.\setup_budget.ps1 -ProjectId "neuralwavesim-prod" -BudgetAmount 10

# デプロイ時に自動でコスト最適化設定
# - min-instances: 0
# - cpu-throttling: enabled
# - timeout: 300s
```

# 監視
```
# コスト状況を確認（週1回推奨）
.\check_costs.ps1 -ProjectId "neuralwavesim-prod"

# または Cloud Console で確認
Start-Process "https://console.cloud.google.com/billing?project=neuralwavesim-prod"
```

# 対応(アラート受信時)
```
# 50% アラート → 監視を強化
.\check_costs.ps1 -ProjectId "neuralwavesim-prod"

# 75% アラート → 使用状況を確認
gcloud run services logs read neuralwavesim-ui --limit 100

# 90% アラート → 緊急停止を検討
.\emergency_stop.ps1 -ProjectId "neuralwavesim-prod"

# 100% アラート → 即座に緊急停止
.\emergency_stop.ps1 -ProjectId "neuralwavesim-prod"
```

#  緊急停止（予算超過時）

```
# すべてのサービスを停止
.\emergency_stop.ps1 -ProjectId "neuralwavesim-prod"

# 確認
gcloud run services list --region asia-northeast1

# 再開（問題解決後）
.\deploy.ps1 -ProjectId "neuralwavesim-prod"
```


