Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Sam2Dir = Join-Path $ProjectRoot "checkpoints\upstream\sam2"
$DinoDir = Join-Path $ProjectRoot "checkpoints\upstream\dinov3"

New-Item -ItemType Directory -Force -Path $Sam2Dir | Out-Null
New-Item -ItemType Directory -Force -Path $DinoDir | Out-Null

function Download-IfMissing {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$OutputPath
    )

    if (Test-Path $OutputPath) {
        Write-Host "skip: $OutputPath already exists"
        return
    }

    Invoke-WebRequest -Uri $Url -OutFile $OutputPath
}

Download-IfMissing `
    -Url "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt" `
    -OutputPath (Join-Path $Sam2Dir "sam2.1_hiera_tiny.pt")

Download-IfMissing `
    -Url "https://dl.fbaipublicfiles.com/dinov3/dinov3_vitb16/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth" `
    -OutputPath (Join-Path $DinoDir "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")

Write-Host "upstream checkpoint download completed"
