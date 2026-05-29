
# VITALScan - How to run the program 

---

## In file explorer
### 1. Make sure you have downloaded the model file to your fileexplorer
###    -  (The model is to large to be added to the repository, has to be shared with a memory stick)
### 3. Make sure you have an MRI Scan you want to diagnose in your fileexplorer
### 2. Make sure that you have a folder including these files:
###     -  The model (best_model.keras)
###     -  translated_data.py
###     -  VitalScan_app.py

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
### 1. Install tensorflow:
   ```bash
   pip install tensorflow
   ```
### 2. Simply run the program with the "run" button or simply in the turminal
###     -  In the terminal:
           ```bash
           python VitalScan_app.py
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
### 1. Find the MRI phtoto you want diagnosed in your file explorer
## Output
### You now get a diagnosis of your uploaded MRI Scan.
   


