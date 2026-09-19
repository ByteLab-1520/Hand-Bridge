$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

function Find-CompatiblePython {
    function Test-PythonCommand($command, $arguments) {
        $previousPreference = $ErrorActionPreference
        try {
            # PowerShell 5 turns stderr from the Microsoft Store alias into a
            # NativeCommandError when the caller uses Stop.
            $ErrorActionPreference = "SilentlyContinue"
            & $command @arguments -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)" *> $null
            return $LASTEXITCODE -eq 0
        }
        finally {
            $ErrorActionPreference = $previousPreference
        }
    }

    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        foreach ($version in @("3.12", "3.11", "3.10")) {
            if (Test-PythonCommand $launcher.Source @("-$version")) {
                return @{ Command = $launcher.Source; Arguments = @("-$version") }
            }
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        if (Test-PythonCommand $python.Source @()) {
            return @{ Command = $python.Source; Arguments = @() }
        }
    }

    return $null
}

Write-Host "Hand-Bridge Windows setup" -ForegroundColor Cyan
$python = Find-CompatiblePython
if (-not $python) {
    Write-Host "Python 3.10-3.12 (64-bit) was not found." -ForegroundColor Red
    Write-Host "Install Python 3.12 (64-bit) from python.org and enable 'Add python.exe to PATH'."
    exit 1
}

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe")) {
    Write-Host "Creating the virtual environment..."
    & $python.Command @($python.Arguments) -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the virtual environment." }
}

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
Write-Host "Installing the validated Windows packages..."
& $venvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip." }
& $venvPython -m pip install -r requirements-windows.txt
if ($LASTEXITCODE -ne 0) { throw "Failed to install packages." }

Write-Host "Validating the installation..."
& $venvPython -c "import cv2, mediapipe, tensorflow as tf; from tensorflow import keras; print('OpenCV', cv2.__version__, '| MediaPipe', mediapipe.__version__, '| TensorFlow', tf.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Installation validation failed." }

Write-Host "Setup complete. Start the app with run_windows.bat." -ForegroundColor Green
