# 古いバージョンの削除
 古いリビジョンをリスト表示
gcloud run revisions list --service neuralwavesim-api --region asia-northeast1

# 古いリビジョンを削除（リビジョン名を指定）
gcloud run revisions delete neuralwavesim-api-00001-abc --region asia-northeast1 --quiet
gcloud run revisions delete neuralwavesim-api-00002-def --region asia-northeast1 --quiet

# UIも同様
gcloud run revisions list --service neuralwavesim-ui --region asia-northeast1
gcloud run revisions delete neuralwavesim-ui-00001-xyz --region asia-northeast1 --quiet


# ビルドテスト
# Docker イメージをビルド
docker-compose build

# ローカルで動作確認
docker-compose up

# ブラウザで確認
# API: http://localhost:8080/docs
# UI:  http://localhost:8501

# 問題なければ停止
docker-compose down


# デプロイ
```
.\deploy.ps1 -ProjectId "neuralwavesim-prod"
```

