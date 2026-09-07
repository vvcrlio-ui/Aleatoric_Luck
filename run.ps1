# Windows convenience entry point; run Linux code under WSL, never emulate locks.
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RunArguments
)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw 'WSL is required for the full engine. Install a Linux distribution first, then retry.'
}
# Pass positional arguments through wsl.exe --exec, without constructing bash -c.
$linuxScript = & wsl.exe --exec wslpath -u (Join-Path $taskRoot 'run.sh')
if ($LASTEXITCODE -ne 0 -or -not $linuxScript) {
    throw 'No usable default WSL distribution. Configure WSL first; no experiment was started.'
}
$linuxArguments = @()
$pathOptions = @('--manifest', '--schema', '--output', '--resume', '--venv')
$previousArgument = ''
foreach ($argument in $RunArguments) {
    if ($previousArgument -in $pathOptions -and $argument -match '^[A-Za-z]:[\\/]') {
        $converted = & wsl.exe --exec wslpath -u $argument
        if ($LASTEXITCODE -ne 0) { throw "Cannot translate path: $argument" }
        $linuxArguments += [string]$converted
    } else {
        $linuxArguments += $argument
    }
    $previousArgument = $argument
}
& wsl.exe --exec bash ([string]$linuxScript) @linuxArguments
exit $LASTEXITCODE
