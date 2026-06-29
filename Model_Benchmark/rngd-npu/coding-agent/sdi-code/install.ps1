# ───────────────────────────────────────────────────────────────────────────
# sdi code 설치기 (Windows PowerShell)
# 서버에 SSH 하지 않고, 내 Windows 에서 서버 NPU LLM 에 붙는 코딩 에이전트 CLI("sdi") 설치.
#
# 사용 (PowerShell). SDI_SERVER = 라우터 주소:
#   원격(집/외부): 먼저 SSH 터널 후 http://127.0.0.1:8400 (외부 입구는 SSH 10022뿐 — 8400 직접 불가)
#       터미널①:  ssh -p 10022 -N -L 8400:localhost:8400 jun@164.125.19.138
#       터미널②:  $env:SDI_SERVER="http://127.0.0.1:8400"   # 인증ON이면 $env:SDI_API_KEY="<키>"
#   사내 LAN: $env:SDI_SERVER="http://10.125.19.138:8400" (사설, 터널 불필요)
#   powershell -ExecutionPolicy Bypass -File install.ps1
#
# 설치 후:  sdi            # 코딩 에이전트 TUI (추론은 서버 NPU)
#           sdi run "..."  # 비대화형
#
# (로직은 실측 통과한 Mac/Linux install.sh 와 동일. Windows 박스에서 'sdi models' 가 서버
#  모델을 돌려주는지 1회 점검 권장.)
# ───────────────────────────────────────────────────────────────────────────
$ErrorActionPreference = "Stop"
if (-not $env:SDI_SERVER) { throw "SDI_SERVER 를 지정하세요 (예: http://10.125.19.138:8400)" }
$server = $env:SDI_SERVER.TrimEnd('/')
$key    = if ($env:SDI_API_KEY) { $env:SDI_API_KEY } else { '' }   # 키는 선택(서버 인증 OFF 면 비움)
if ($key -and ($key -notmatch '^[A-Za-z0-9._-]+$')) { throw "SDI_API_KEY 에 허용되지 않는 문자(영숫자와 . _ - 만 허용)" }
$cmd   = if ($env:SDI_CMD)   { $env:SDI_CMD }   else { 'sdi' }                 # 명령(=provider) 이름 — 리브랜딩 시 SDI_CMD 로 변경
$brand = if ($env:SDI_BRAND) { $env:SDI_BRAND } else { 'SDI Code (Furiosa NPU)' }

$SdiHome = Join-Path $env:USERPROFILE ".$cmd"
$BinDir  = Join-Path $SdiHome "bin"
$Cfg     = Join-Path $SdiHome "opencode.json"
New-Item -ItemType Directory -Force -Path $SdiHome,$BinDir | Out-Null

Write-Host "[1/3] opencode 설치 확인"
if (-not (Get-Command opencode -ErrorAction SilentlyContinue)) {
  Write-Host "      설치 시도 (npm → scoop → choco 순)"
  if     (Get-Command npm   -ErrorAction SilentlyContinue) { npm install -g opencode-ai }
  elseif (Get-Command scoop -ErrorAction SilentlyContinue) { scoop install opencode }
  elseif (Get-Command choco -ErrorAction SilentlyContinue) { choco install opencode -y }
  else { throw "opencode 자동설치 실패. https://opencode.ai/docs 보고 수동 설치 후 다시 실행하세요." }
}

Write-Host "[2/3] 서버에서 모델 목록 받아 sdi 설정 생성(키 포함): $server"
$headers = if ($key) { @{ Authorization = "Bearer $key" } } else { @{} }
try {
  $resp = Invoke-RestMethod -Uri "$server/v1/models" -Headers $headers -TimeoutSec 10
} catch {
  $status = $null; try { $status = [int]$_.Exception.Response.StatusCode } catch {}
  if ($status -eq 401) {
    Write-Host "[fail] 서버 응답 오류: HTTP 401 = 서버 인증 ON. 발급받은 키로 재실행:"
    Write-Host '        $env:SDI_API_KEY="<키>"; powershell -ExecutionPolicy Bypass -File install.ps1'
  } else {
    $u = [Uri]$server; $h = $u.Host; $p = $u.Port
    $priv = $h -match '^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)'
    Write-Host "[fail] 서버 도달 실패 $server/v1/models : $($_.Exception.Message)"
    Write-Host "       └ IP·포트를 안 넣어서가 아니라(이미 SDI_SERVER 로 지정됨) '네트워크 경로'가 없을 때 납니다."
    Write-Host "       └ 점검: 이 PC가 서버와 같은 망인가요? $h 가 사내 사설 IP($(if($priv){'그렇습니다'}else{'확인필요'}))면 외부에선 VPN/사내망 연결이 필요합니다."
    Write-Host "       └ 빠른 확인:  Test-NetConnection $h -Port $p   (TcpTestSucceeded=True → 재시도 / False → 망문제(VPN) 또는 서버측 점검)"
  }
  exit 1
}
$ids = @($resp.data | ForEach-Object { $_.id })
if ($ids.Count -eq 0) { throw "서버에 모델이 없습니다 ($server/v1/models)" }
function Get-Ctx($m) { $l = $m.ToLower(); if ($l -match '16k') { 16384 } elseif ($l -match 'a3b') { 65536 } else { 32768 } }
$models = [ordered]@{}
foreach ($m in $ids) { $models[$m] = @{ name = $m; limit = @{ context = (Get-Ctx $m); output = 8192 } } }
$default = if ($ids -contains 'Qwen3-32B-FP8') { 'Qwen3-32B-FP8' } else { $ids[0] }
$opts = @{ baseURL = "$server/v1" }
if ($key) { $opts.apiKey = $key }                            # 키 없으면 apiKey 생략(무인증). 키는 설정파일에만.
$prov = @{}
$prov[$cmd] = @{
  npm     = '@ai-sdk/openai-compatible'
  name    = $brand
  options = $opts
  models  = $models
}
$cfg = [ordered]@{
  '$schema'   = 'https://opencode.ai/config.json'
  provider    = $prov
  model       = "$cmd/$default"
  small_model = "$cmd/$default"
}
$json = $cfg | ConvertTo-Json -Depth 10
# UTF-8 BOM 없이 기록(PS 5.1 의 -Encoding UTF8 은 BOM 을 붙여 JSON 파서가 깨질 수 있음)
[System.IO.File]::WriteAllText($Cfg, $json, (New-Object System.Text.UTF8Encoding($false)))
# 설정파일(키 포함)을 현재 사용자 전용으로 잠금
icacls $Cfg /inheritance:r /grant:r "$($env:USERNAME):(R,W)" | Out-Null
Write-Host "[ok] 모델 $($ids.Count)개 등록 (기본 $default)"

Write-Host "[3/3] '$cmd' 명령 설치: $BinDir\$cmd.cmd"
# 래퍼는 OPENCODE_CONFIG(문서화된 크로스플랫폼 설정경로)만 가리킨다. 키/비밀은 cmd 에 없음.
$cmdBody = "@echo off`r`nset `"OPENCODE_CONFIG=$Cfg`"`r`nopencode %*`r`n"
[System.IO.File]::WriteAllText((Join-Path $BinDir "$cmd.cmd"), $cmdBody, (New-Object System.Text.ASCIIEncoding))

Write-Host "[ ] PATH 에 $BinDir 추가 (사용자 환경변수, 세그먼트 단위 검사)"
$userPath = [Environment]::GetEnvironmentVariable("Path","User")
$segments = @($userPath -split ';' | Where-Object { $_ -ne '' })
if ($segments -notcontains $BinDir) {
  [Environment]::SetEnvironmentVariable("Path", (@($BinDir) + $segments -join ';'), "User")
  Write-Host "   ⚠️ 새 터미널을 열어야 PATH 가 반영됩니다."
}
Write-Host ""
Write-Host "[OK] 설치 완료. (설정·키: $Cfg, 사용자 전용 잠금)"
Write-Host "     실행:  $cmd   /   $cmd run ""..."" /   $cmd models   /   $cmd agent list"
Write-Host "     제거:  Remove-Item -Recurse -Force ""$SdiHome""   (키 회전 시 재실행)"
