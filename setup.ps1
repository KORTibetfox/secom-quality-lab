$ErrorActionPreference='Stop'
$projectRoot=$PSScriptRoot
$bundledPython=Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$envDir=Join-Path $projectRoot '.venv'
$venvPython=Join-Path $envDir 'Scripts/python.exe'
New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot 'tmp') | Out-Null
$env:TEMP=Join-Path $projectRoot 'tmp'
$env:TMP=$env:TEMP
if(Test-Path -LiteralPath $bundledPython){
  & $bundledPython -m venv --without-pip $envDir
  if($LASTEXITCODE -ne 0){throw 'Failed to create virtual environment'}
  $lock=Join-Path $projectRoot 'requirements-lock.txt'
  if(!(Test-Path -LiteralPath $lock)){$lock=Join-Path $projectRoot 'requirements.txt'}
  & $bundledPython -m pip --python $venvPython install --cache-dir (Join-Path $projectRoot '.cache/pip') -r $lock
} else {
  py -3.12 -m venv $envDir
  if($LASTEXITCODE -ne 0){throw 'Install Python 3.12 or use the Codex bundled Python runtime.'}
  & $venvPython -m pip install -r (Join-Path $projectRoot 'requirements-lock.txt')
}
if($LASTEXITCODE -ne 0){throw 'Dependency installation failed'}
Write-Host 'Setup complete. Run ./run.ps1 -Action test, then -Action train, then -Action notebook.'
