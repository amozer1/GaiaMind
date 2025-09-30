@echo off
REM -------------------------------
REM GaiaMind Setup Script for Windows
REM -------------------------------

REM Step 1: Navigate to Users folder
cd C:\Users\adane

REM Step 2: Create project folder
if not exist GaiaMind mkdir GaiaMind
cd GaiaMind

REM Step 3: Create folder structure
mkdir data notebooks src tests
mkdir src\ml_models src\preprocessing src\utils src\app
mkdir data\satellite data\sensors data\social_news

REM Step 4: Create virtual environment
python -m venv venv

REM Step 5: Activate virtual environment
call venv\Scripts\activate

REM Step 6: Upgrade pip
python -m pip install --upgrade pip

REM Step 7: Create requirements.txt
echo numpy>requirements.txt
echo pandas>>requirements.txt
echo scikit-learn>>requirements.txt
echo matplotlib>>requirements.txt
echo seaborn>>requirements.txt
echo torch>>requirements.txt
echo torchvision>>requirements.txt
echo transformers>>requirements.txt
echo tensorflow>>requirements.txt
echo fastapi>>requirements.txt
echo flask>>requirements.txt
echo uvicorn>>requirements.txt
echo psycopg2-binary>>requirements.txt
echo sqlalchemy>>requirements.txt
echo redis>>requirements.txt
echo plotly>>requirements.txt
echo dash>>requirements.txt
echo requests>>requirements.txt
echo geopandas>>requirements.txt
echo folium>>requirements.txt

REM Step 8: Install Python packages
pip install -r requirements.txt

echo.
echo --------------------------------
echo GaiaMind setup complete!
echo Virtual environment is active. Use "deactivate" to exit it.
echo Folder structure and packages are ready at C:\Users\adane\GaiaMind
echo --------------------------------
pause
