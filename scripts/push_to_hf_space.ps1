param(
    [Parameter(Mandatory = $true)]
    [string]$SpaceRepo,

    [string]$WorkDir = "./hf-space"
)

# Example:
#   ./scripts/push_to_hf_space.ps1 -SpaceRepo "https://huggingface.co/spaces/Dhairyagothi/RouteBrain"

$ErrorActionPreference = "Stop"

if (Test-Path $WorkDir) {
    Remove-Item -Recurse -Force $WorkDir
}

Write-Host "Cloning Space repository..."
git clone $SpaceRepo $WorkDir

Write-Host "Copying project files..."
Get-ChildItem -Force . | Where-Object {
    $_.Name -notin @(".git", "hf-space")
} | ForEach-Object {
    Copy-Item -Recurse -Force $_.FullName (Join-Path $WorkDir $_.Name)
}

Push-Location $WorkDir

Write-Host "Committing and pushing..."
git add .
if (git status --porcelain) {
    git commit -m "Deploy RouteBrain Space"
    git push
} else {
    Write-Host "No changes to push."
}

Pop-Location
Write-Host "Done."
