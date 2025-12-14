# 1. ベースイメージ
FROM python:3.11-slim

# 2. 作業ディレクトリ
WORKDIR /app

# 3. システム依存関係のインストール
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. 依存関係ファイルのコピー
COPY requirements.txt .

# --- 変更点: タイムアウト時間を伸ばし、キャッシュを使わずインストール ---
# --default-timeout=1000: ネットワークが遅くても諦めないように設定
# pip install の分割はあえてせず、タイムアウト延長で対応します
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# 5. アプリコードのコピー
COPY . .

# 6. ポート設定（警告が出ていたので "=" をつけました）
ENV PORT=8080

# 7. モデルディレクトリの作成（事前学習済みモデル用）
RUN mkdir -p models

# 8. ヘルスチェック
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# 9. 起動コマンド
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]