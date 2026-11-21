@echo off
title Push Synthetic Dataset Generator to GitHub
echo =========================================================================
echo   Pushing code to https://github.com/Shershah09/synthetic-dataset-generator
echo =========================================================================
echo.
git push -u origin main
echo.
if %ERRORLEVEL% EQU 0 (
    echo =========================================================================
    echo   SUCCESS: Code pushed to GitHub!
    echo   Now go to https://share.streamlit.io to deploy your app in 1 click!
    echo =========================================================================
) else (
    echo.
    echo If you see 'Repository not found', please make sure you created the repo:
    echo https://github.com/new?name=synthetic-dataset-generator
)
echo.
pause
