# ───────────────────────────────────────────────────────────────────────────
# furio 설치기 (Windows PowerShell) — Claude Code 같은 코딩 에이전트(openclaude) + 서버 NPU
#
# 사용 (PowerShell). SDI_SERVER = 라우터 주소:
#   원격(집/외부): 먼저 SSH 터널 후 http://127.0.0.1:8400 (외부 입구는 SSH 10022뿐)
#       터미널①:  ssh -p 10022 -N -L 8400:localhost:8400 jun@164.125.19.138
#       터미널②:  $env:SDI_SERVER="http://127.0.0.1:8400"   # 인증ON이면 $env:SDI_API_KEY="<키>"
#   사내 LAN: $env:SDI_SERVER="http://10.125.19.138:8400"
#   powershell -ExecutionPolicy Bypass -File install.ps1
#
# 설치 후:  furio            # Claude 같은 코딩 에이전트 TUI
#           furio -p "..."   # 비대화형 한 줄
# ───────────────────────────────────────────────────────────────────────────
$ErrorActionPreference = "Stop"
if (-not $env:SDI_SERVER) { throw "SDI_SERVER 를 지정하세요 (예: http://127.0.0.1:8400)" }
$server = $env:SDI_SERVER.TrimEnd('/')
$key    = if ($env:SDI_API_KEY) { $env:SDI_API_KEY } else { '' }
if ($key -and ($key -notmatch '^[A-Za-z0-9._-]+$')) { throw "SDI_API_KEY 에 허용 안 되는 문자(영숫자와 . _ - 만)" }
$cmd    = if ($env:FURIO_CMD)        { $env:FURIO_CMD }        else { 'furio' }
$model  = if ($env:FURIO_MODEL)      { $env:FURIO_MODEL }      else { 'gpt-oss-120b' }
$maxout = if ($env:FURIO_MAX_OUTPUT) { $env:FURIO_MAX_OUTPUT } else { '8192' }
$auto   = if ($env:FURIO_AUTO)       { $env:FURIO_AUTO }       else { '' }   # 완전자동 기본값(빈값=확인모드)
$HomeDir = Join-Path $env:USERPROFILE ".$cmd"
$BinDir  = Join-Path $HomeDir "bin"
New-Item -ItemType Directory -Force -Path $HomeDir,$BinDir | Out-Null

Write-Host "[1/4] Node >=22 확인 (openclaude 요구)"
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw "node 없음 — Node >=22 설치 후 재실행 (https://nodejs.org)" }
$nodeMajor = [int](node -p "process.versions.node.split('.')[0]")
if ($nodeMajor -lt 22) { throw "node $(node -v) — openclaude 는 Node >=22 필요. 최신 Node 설치 후 재실행." }

Write-Host "[2/4] openclaude 설치 (격리 prefix: $HomeDir)"
npm install -g '@gitlawb/openclaude@latest' --prefix $HomeDir | Out-Null
$ocBin = Join-Path $HomeDir "openclaude.cmd"     # Windows npm --prefix 는 prefix 루트에 .cmd 생성
if (-not (Test-Path $ocBin)) { $ocBin = Join-Path $BinDir "openclaude.cmd" }
if (-not (Test-Path $ocBin)) { throw "openclaude 바이너리 없음 ($HomeDir)" }

Write-Host "[3/4] 서버 도달 확인: $server"
$headers = if ($key) { @{ Authorization = "Bearer $key" } } else { @{} }
try { Invoke-RestMethod -Uri "$server/v1/models" -Headers $headers -TimeoutSec 10 | Out-Null; Write-Host "      [ok] /v1/models 응답" }
catch { Write-Host "[fail] 서버 도달 실패 $server/v1/models — SSH 터널이 떠 있나요? (ssh -p 10022 -N -L 8400:localhost:8400 ...)"; exit 1 }

# 키는 파일에(있을 때) + 사용자 전용 잠금. 래퍼 텍스트엔 비밀 없음.
$keyFile = Join-Path $HomeDir "key"
if ($key) { [System.IO.File]::WriteAllText($keyFile, $key, (New-Object System.Text.ASCIIEncoding)); icacls $keyFile /inheritance:r /grant:r "$($env:USERNAME):(R,W)" | Out-Null }
elseif (Test-Path $keyFile) { Remove-Item $keyFile }

Write-Host "[4/4] '$cmd' 명령 설치: $BinDir\$cmd.cmd"
$lines = @(
  '@echo off',
  'set "CLAUDE_CODE_USE_OPENAI=1"',
  "set `"OPENAI_BASE_URL=$server/v1`"",
  "if not defined OPENAI_MODEL set `"OPENAI_MODEL=$model`"",
  "if not defined CLAUDE_CODE_MAX_OUTPUT_TOKENS set `"CLAUDE_CODE_MAX_OUTPUT_TOKENS=$maxout`"",
  "set `"OPENCLAUDE_CONFIG_DIR=$HomeDir\config`"",
  'set "OPENAI_API_KEY=dummy"',
  "if exist `"$keyFile`" set /p OPENAI_API_KEY=<`"$keyFile`"",
  "if not defined FURIO_AUTO set `"FURIO_AUTO=$auto`"",
  'set "AUTO_ARGS="',
  'if /I "%FURIO_AUTO%"=="1" set "AUTO_ARGS=--dangerously-skip-permissions"',
  'if /I "%FURIO_AUTO%"=="yes" set "AUTO_ARGS=--dangerously-skip-permissions"',
  'if /I "%FURIO_AUTO%"=="on" set "AUTO_ARGS=--dangerously-skip-permissions"',
  'if /I "%FURIO_AUTO%"=="full" set "AUTO_ARGS=--dangerously-skip-permissions"',
  'if /I "%FURIO_AUTO%"=="bypass" set "AUTO_ARGS=--dangerously-skip-permissions"',
  'if /I "%FURIO_AUTO%"=="edits" set "AUTO_ARGS=--permission-mode acceptEdits"',
  'if /I "%FURIO_AUTO%"=="accept" set "AUTO_ARGS=--permission-mode acceptEdits"',
  "`"$ocBin`" %AUTO_ARGS% %*"
) -join "`r`n"
[System.IO.File]::WriteAllText((Join-Path $BinDir "$cmd.cmd"), $lines + "`r`n", (New-Object System.Text.ASCIIEncoding))

# PATH 에 $BinDir 추가 (사용자 환경변수, 세그먼트 단위)
$userPath = [Environment]::GetEnvironmentVariable("Path","User")
$segments = @($userPath -split ';' | Where-Object { $_ -ne '' })
if ($segments -notcontains $BinDir) {
  [Environment]::SetEnvironmentVariable("Path", (@($BinDir) + $segments -join ';'), "User")
  Write-Host "   ⚠️ 새 터미널을 열어야 PATH 가 반영됩니다."
}
Write-Host ""
Write-Host "[OK] 설치 완료. (openclaude: $HomeDir, 명령: $BinDir\$cmd.cmd)"
Write-Host "     실행:  $cmd   /   $cmd -p ""..."" /   $cmd --model gpt-oss-120b"
Write-Host "     ⚠️ 작업은 일반 프로젝트 폴더에서(.claude 등 민감 경로엔 쓰기 차단). 터널 방식이면 furio 쓰는 동안 터널 유지."
Write-Host "     제거:  Remove-Item -Recurse -Force ""$HomeDir""; Remove-Item ""$BinDir\$cmd.cmd"""
