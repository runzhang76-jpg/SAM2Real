Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExternalDir = Join-Path $ProjectRoot "external"

New-Item -ItemType Directory -Force -Path $ExternalDir | Out-Null

function Clone-IfMissing {
    param(
        [Parameter(Mandatory = $true)][string]$RepoUrl,
        [Parameter(Mandatory = $true)][string]$TargetDir
    )

    $gitDir = Join-Path $TargetDir ".git"
    if (Test-Path $gitDir) {
        Write-Host "skip: $TargetDir already exists"
        return
    }

    if (Test-Path $TargetDir) {
        throw "error: $TargetDir exists but is not a git repository"
    }

    git clone $RepoUrl $TargetDir
}

Clone-IfMissing -RepoUrl "https://github.com/facebookresearch/sam2.git" -TargetDir (Join-Path $ExternalDir "sam2")
Clone-IfMissing -RepoUrl "https://github.com/facebookresearch/dinov3.git" -TargetDir (Join-Path $ExternalDir "dinov3")

Write-Host "upstream source download completed"
