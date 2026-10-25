param([ValidateSet('train','diagnose','notebook','test','verify','demo','release','serve','api-demo','smoke')][string]$Action='demo')
$ErrorActionPreference='Stop'
$projectRoot=$PSScriptRoot
$projectPython=Join-Path $projectRoot '.venv/Scripts/python.exe'
$sessionPython=Join-Path $projectRoot '../../work/.venv/Scripts/python.exe'
if(Test-Path -LiteralPath $projectPython){$runner=$projectPython}
elseif(Test-Path -LiteralPath $sessionPython){$runner=$sessionPython}
else{throw 'Python environment not found. Run setup.ps1 first.'}
Push-Location $projectRoot
try {
  New-Item -ItemType Directory -Force -Path 'tmp','.cache' | Out-Null
  $env:TEMP=Join-Path $projectRoot 'tmp'
  $env:TMP=$env:TEMP
  $env:MPLCONFIGDIR=Join-Path $projectRoot '.cache/matplotlib'
  if($Action -eq 'train'){ & $runner 'src/train.py' }
  elseif($Action -eq 'diagnose'){ & $runner 'src/diagnose.py' }
  elseif($Action -eq 'notebook'){ & $runner 'src/build_notebook.py' }
  elseif($Action -eq 'test'){ & $runner -m unittest discover -s tests -v }
  elseif($Action -eq 'verify'){ & $runner 'src/verify_results.py' }
  elseif($Action -eq 'release'){ & $runner 'src/release.py' }
  elseif($Action -eq 'serve'){ & $runner 'src/serve.py' }
  elseif($Action -eq 'api-demo'){ & $runner 'src/client.py' }
  elseif($Action -eq 'smoke'){ & $runner 'src/smoke.py' }
  else{ & $runner 'src/predict.py' --input 'data/demo_sensors.csv' --output 'results/demo_ranking.csv' --budget .1 }
  if($LASTEXITCODE -ne 0){throw "Command failed with code $LASTEXITCODE"}
} finally {Pop-Location}
