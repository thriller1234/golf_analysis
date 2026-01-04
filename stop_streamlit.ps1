# Streamlitアプリを停止するスクリプト

Write-Host "Streamlitプロセスを検索中..." -ForegroundColor Yellow

# ポート8501を使用しているプロセスを検索
$portProcess = netstat -ano | findstr :8501 | Select-String "LISTENING"

if ($portProcess) {
    $processId = ($portProcess -split '\s+')[-1]
    Write-Host "プロセスID $processId を終了します..." -ForegroundColor Yellow
    
    try {
        Stop-Process -Id $processId -Force
        Write-Host "✅ Streamlitプロセスを終了しました" -ForegroundColor Green
    } catch {
        Write-Host "❌ エラー: プロセスを終了できませんでした" -ForegroundColor Red
        Write-Host $_.Exception.Message
    }
} else {
    Write-Host "ポート8501を使用しているプロセスが見つかりませんでした" -ForegroundColor Yellow
}

# Pythonプロセスも確認
Write-Host "`n残っているPythonプロセス:" -ForegroundColor Yellow
Get-Process | Where-Object {$_.ProcessName -like "*python*"} | Select-Object Id, ProcessName, CPU | Format-Table

