# Streamlitアプリを起動するスクリプト

Write-Host "ゴルフスイング解析アプリを起動します..." -ForegroundColor Green
Write-Host ""

# エラーハンドリングを追加
try {
    # Streamlitアプリを起動
    streamlit run app.py --server.headless false
} catch {
    Write-Host "エラーが発生しました: $_" -ForegroundColor Red
    exit 1
} finally {
    Write-Host "`nアプリを終了しました。" -ForegroundColor Yellow
}

