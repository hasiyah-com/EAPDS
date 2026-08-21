@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo =====================================================
echo   Push EAPDS -^> https://github.com/hasiyah-com/EAPDS
echo   โฟลเดอร์: %CD%
echo =====================================================
echo.

REM --- ตรวจว่ามี git ---
git --version >nul 2>&1
if errorlevel 1 (
  echo [ERROR] ยังไม่ได้ติดตั้ง git - ติดตั้งจาก https://git-scm.com ก่อน
  pause
  exit /b 1
)

REM --- ตั้งค่า identity เฉพาะ repo นี้ ---
git init
git config user.email "hasiyahdama5@gmail.com"
git config user.name "hasiyah-com"
git branch -M main

REM --- ตรวจก่อนว่าไม่มีความลับหลุด ---
echo.
echo กำลังจะเพิ่มไฟล์เหล่านี้ (ต้องไม่มี firebase_key.json):
git add .
git status --short
echo.

git commit -m "Initial commit: EAPDS phishing detection system (code + small models)"

git remote remove origin >nul 2>&1
git remote add origin https://github.com/hasiyah-com/EAPDS.git

echo.
echo กำลัง push ... (อาจมีหน้าต่างให้ล็อกอิน GitHub)
git push -u origin main

echo.
echo ถ้า push ไม่ผ่านเพราะ repo มีไฟล์อยู่แล้ว (เช่นมี README บน GitHub)
echo ให้รันคำสั่งนี้: git pull --rebase origin main  แล้ว git push -u origin main อีกครั้ง
echo.
pause
