# 1. ベースイメージ
FROM python:3.10-slim

# 2. 作業ディレクトリ
WORKDIR /code

# 3. 依存関係ファイルのコピー
COPY ./requirements.txt /code/requirements.txt

# --- 変更点: タイムアウト時間を伸ばし、キャッシュを使わずインストール ---
# --default-timeout=1000: ネットワークが遅くても諦めないように設定
# pip install の分割はあえてせず、タイムアウト延長で対応します
RUN pip install --default-timeout=1000 --no-cache-dir --upgrade -r /code/requirements.txt

# 4. アプリコードのコピー
COPY ./app /code/app

# 追記: PYTHONPATH 環境変数の設定
ENV PYTHONPATH=/code/app
# 5. ポート設定（警告が出ていたので "=" をつけました）
ENV PORT=8080

# 6. 起動コマンド
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]