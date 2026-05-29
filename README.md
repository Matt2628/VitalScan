
# VITALScan - How to run the program 

---

## In file explorer
### Make sure that you have unpacked the file if it is zipped.
### The files needed for the program are:
#### The model: best_model.keras
#### MRI Scan: MRI Scan - Meniglioma
##### (These files are stores in the "VitalScan.main" folder in the "project_code" folder)

&nbsp;
## Virtual environment
#### 1. Create your virtual environment
```bash
python -m venv venv
````
#### 2. Activate your virtual environment
##### Windows:
```bash
venv\Scripts\activate
````
##### Mac/Linux:
```bash
source venv/bin/activate
````

&nbsp;
## Programs that need to be installed
#### (Make sure that these are installed inside the venv)
#### If any program recommends upgrading any of the software, so as such.
### 1. TensorFlow:
   ```bash
   pip install tensorflow
   ```
### 2. Pillow:
   ```bash
   pip install pillow
   ```

&nbsp;
## Running the code
### Access the correct folder.
#### When opening the program in python you are directed to the "Vitalscan.main" folder
#### The code are stores in the "Ai.program" folder.
#### Access this folder by writing this in the terminal:
Skriv dette i terminalen:
```bash
cd "Ai.program"
python Vitalscan.py
```
### Run the code from the Vitalscan.py file
#### Press the "play" button in the top right of the screen (python)
#### Or run the code directly in the terminal:
```bash
python Vitalscan.py
```

&nbsp;
# VITALScan - How to use the program

---

## Inputs
### 1. Write the name of your patient 
### 2. Write the date of the procedure
## The model
### 1. Find the model used earlier in your file explorer (best_model.keras)
## MRI Scan 
### 1. Find the MRI image you want diagnosed in your file explorer (MRI Scan - Meniglioma)
## Output
### You now get a diagnosis of your uploaded MRI Scan.
   


