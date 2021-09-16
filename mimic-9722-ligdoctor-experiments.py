import sys
import subprocess
from contextlib import redirect_stdout

for r in range(3):
  # cell [542]
  stdout = subprocess.check_output('python2.7 src/LigDoctor-latest/LIG-Doctor.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/lig-doctor-542 --hiddenDimSize=[542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python2.7 src/LigDoctor-latest/LIG-Doctor-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/lig-doctor-542 --hiddenDimSize=[542]', shell=True)

  with open(str(r) + '_ligdoctor-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[542]")
      print(stdout.decode().split("\n"))
      print("\n")


for r in range(3):
  # cell [271]
  stdout = subprocess.check_output('python2.7 src/LigDoctor-latest/LIG-Doctor.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/lig-doctor-271 --hiddenDimSize=[271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python2.7 src/LigDoctor-latest/LIG-Doctor-test.py "../clinical-prediction/data/mimic-ed/both/mimic-ed_90-10_9722" compiled_models/lig-doctor-271 --hiddenDimSize=[271]', shell=True)

  with open(str(r) + '_ligdoctor-mimic-9722-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print(" --hiddenDimSize=[271]")
      print(stdout.decode().split("\n"))
      print("\n")
