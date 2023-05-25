# Viveka AI Hackathon
This project aims to detect frogeris in the news documents. The project is based on the [FRAUD DETECTION CONTEST : FIND IT !](http://findit.univ-lr.fr/). The project is a part of the Viveka AI Hackathon.

## Usage
- Clone the repository
```bash
git clone https://github.com/SrjPdl/viveka-hackathon.git
```
- Create a virtual environment using anaconda or python venv
```bash
conda create -n env python=3.9
```
```bash
python3 -m venv env
```
- Activate the environment
```bash
conda activate env
```
```bash
source env/bin/activate
```
- Install the requirements
```bash
pip install -r requirements.txt
```
- Train the model
```bash
python3 src/pipelines/train_pipeline.py
```
- Run the web app
```bash
cd src/app
uvicorn main:app --reload
```
- Open the web app in the browser and go to url for the docs and test the API
```url
http://127.0.0.1:8000/docs 
```