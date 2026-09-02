@echo off
setlocal
cd /d "%~dp0.."

set "DEST=root@101.79.20.171:/srv/festival/app"

echo === Full app sync (feat/qa-driven-routing, no git pull) ===
echo Local: %CD%
echo This replaces server app/ code to match your laptop.
echo.

echo [1/3] app/  (entire Python package)
scp -r "%CD%\app" "%DEST%/"

echo [2/4] scripts/
scp "%CD%\scripts\run_peer_compare_batch.py" "%DEST%/scripts/run_peer_compare_batch.py"

echo [3/4] qa-tool/examples/peer-compare*
scp "%CD%\qa-tool\examples\peer-compare-batch.txt" "%DEST%/qa-tool/examples/peer-compare-batch.txt"
scp -r "%CD%\qa-tool\examples\peer-compare" "%DEST%/qa-tool/examples/"

echo [4/4] qa-tool/examples/complex-queries*
scp "%CD%\qa-tool\examples\complex-queries-batch.txt" "%DEST%/qa-tool/examples/complex-queries-batch.txt"
scp -r "%CD%\qa-tool\examples\complex-queries" "%DEST%/qa-tool/examples/"

echo.
echo === Optional: remove server-only orphan (if import errors persist) ===
echo   ssh root@101.79.20.171 "rm -f /srv/festival/app/app/reasoning/clarification_candidates.py"
echo.
echo === Run E2E on server ===
echo   cd /srv/festival/app
echo   source .venv/bin/activate
echo   set -a ^&^& source .env ^&^& set +a
echo   FESTIVAL_HCX_ENABLED=false python3 scripts/run_peer_compare_batch.py --mode e2e
endlocal
