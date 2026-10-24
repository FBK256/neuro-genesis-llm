[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Model,
    [Parameter(Mandatory = $true)]
    [string]$Mmproj,
    [string]$Server = "llama-server",
    [string]$Alias = "local-vlm",
    [ValidateRange(1, 65535)]
    [int]$Port = 8080,
    [ValidateRange(1024, 131072)]
    [int]$ContextSize = 8192
)

$ErrorActionPreference = "Stop"

# Resolve and validate local model paths before starting the listener.
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "Model file not found: $Model"
}
if (-not (Test-Path -LiteralPath $Mmproj -PathType Leaf)) {
    throw "mmproj file not found: $Mmproj"
}
$ModelPath = (Resolve-Path -LiteralPath $Model).Path
$MmprojPath = (Resolve-Path -LiteralPath $Mmproj).Path
$ServerCommand = Get-Command $Server -ErrorAction SilentlyContinue
if ($null -eq $ServerCommand) {
    throw "llama-server was not found. Pass -Server with its executable path."
}

Write-Host "Starting a loopback-only llama.cpp server."
Write-Host "Model: $ModelPath"
Write-Host "mmproj: $MmprojPath"
Write-Host "Endpoint: http://127.0.0.1:$Port/v1"
Write-Host "No shell/browser/MCP tools are enabled by this launcher."
Write-Host "Use an mmproj built for this exact model family/revision."

# Deliberately bind to loopback and never pass --tools or an externally reachable host.
$ServerArgs = @(
    "-m", $ModelPath,
    "--mmproj", $MmprojPath,
    "--alias", $Alias,
    "--host", "127.0.0.1",
    "--port", "$Port",
    "--ctx-size", "$ContextSize",
    "--parallel", "1"
)
& $ServerCommand.Source @ServerArgs

if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
