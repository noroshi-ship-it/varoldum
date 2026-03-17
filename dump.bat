@echo off
setlocal enabledelayedexpansion

set "OUTPUT=all.txt"

if exist "%OUTPUT%" del "%OUTPUT%"

echo ===============================================>>"%OUTPUT%"
echo Python source dump>>"%OUTPUT%"
echo Generated at %date% %time%>>"%OUTPUT%"
echo Root: %cd%>>"%OUTPUT%"
echo ===============================================>>"%OUTPUT%"
echo.>>"%OUTPUT%"

for /r %%F in (*.py) do (
    echo ===============================================>>"%OUTPUT%"
    echo FILE: %%F>>"%OUTPUT%"
    echo ===============================================>>"%OUTPUT%"
    type "%%F">>"%OUTPUT%"
    echo.>>"%OUTPUT%"
    echo.>>"%OUTPUT%"
)

echo Done. Output written to %OUTPUT%
pause