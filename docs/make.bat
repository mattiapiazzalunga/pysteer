@ECHO OFF

pushd "%~dp0"
set EXITCODE=0
set SOURCEDIR=.
set BUILDDIR=_build

if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=python -m sphinx
)

if "%1" == "" goto help

if "%1" == "open" (
    %SPHINXBUILD% -M html "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
    if errorlevel 1 (
        set EXITCODE=1
        goto end
    )
    python -c "from pathlib import Path; import webbrowser; p = Path(r'%BUILDDIR%/html/index.html').resolve(); print(p); webbrowser.open(p.as_uri())"
    if errorlevel 1 set EXITCODE=1
    goto end
)

%SPHINXBUILD% -M %1 "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
if errorlevel 1 set EXITCODE=1
goto end

:help
%SPHINXBUILD% -M help "%SOURCEDIR%" "%BUILDDIR%" %SPHINXOPTS% %O%
if errorlevel 1 set EXITCODE=1

:end
popd
exit /b %EXITCODE%
