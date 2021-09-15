import sys
import subprocess
from contextlib import redirect_stdout

for r in range(5):
  # cell [271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=25 --hiddenDimSize=[271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=25 --hiddenDimSize=[271] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + 'encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=25 --hiddenDimSize=[271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [271, 271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=25 --hiddenDimSize=[271,271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=25 --hiddenDimSize=[271,271] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=25 --hiddenDimSize=[271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [271, 271, 271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=25 --hiddenDimSize=[271,271,271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=25 --hiddenDimSize=[271,271,271] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=25 --hiddenDimSize=[271, 271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=25 --hiddenDimSize=[542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=25 --hiddenDimSize=[542] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=25 --hiddenDimSize=[542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [542, 542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=25 --hiddenDimSize=[542,542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=25 --hiddenDimSize=[542,542] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=25 --hiddenDimSize=[542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [542, 542, 542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=25 --hiddenDimSize=[542,542,542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=25 --hiddenDimSize=[542,542,542] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=25 --hiddenDimSize=[542, 542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1084]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=25 --hiddenDimSize=[1084]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=25 --hiddenDimSize=[1084] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=25 --hiddenDimSize=[1084]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1084, 1084]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=25 --hiddenDimSize=[1084,1084]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=25 --hiddenDimSize=[1084,1084] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=25 --hiddenDimSize=[1084, 1084]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1084, 1084, 1084]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=25 --hiddenDimSize=[1084,1084,1084]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=25 --hiddenDimSize=[1084,1084,1084] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=25 --hiddenDimSize=[1084, 1084, 1084]")
      print(stdout.decode().split("\n"))
      print("\n")



  # cell [271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=50 --hiddenDimSize=[271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=50 --hiddenDimSize=[271] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + 'encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=50 --hiddenDimSize=[271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [271, 271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=50 --hiddenDimSize=[271,271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=50 --hiddenDimSize=[271,271] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=50 --hiddenDimSize=[271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [271, 271, 271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=50 --hiddenDimSize=[271,271,271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=50 --hiddenDimSize=[271,271,271] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=50 --hiddenDimSize=[271, 271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=50 --hiddenDimSize=[542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=50 --hiddenDimSize=[542] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=50 --hiddenDimSize=[542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [542, 542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=50 --hiddenDimSize=[542,542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=50 --hiddenDimSize=[542,542] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=50 --hiddenDimSize=[542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [542, 542, 542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=50 --hiddenDimSize=[542,542,542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=50 --hiddenDimSize=[542,542,542] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=50 --hiddenDimSize=[542, 542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1084]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=50 --hiddenDimSize=[1084]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=50 --hiddenDimSize=[1084] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=50 --hiddenDimSize=[1084]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1084, 1084]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=50 --hiddenDimSize=[1084,1084]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=50 --hiddenDimSize=[1084,1084] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=50 --hiddenDimSize=[1084, 1084]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1084, 1084, 1084]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=50 --hiddenDimSize=[1084,1084,1084]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=50 --hiddenDimSize=[1084,1084,1084] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=50 --hiddenDimSize=[1084, 1084, 1084]")
      print(stdout.decode().split("\n"))
      print("\n")



  # cell [271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=100 --hiddenDimSize=[271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=100 --hiddenDimSize=[271] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + 'encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=100 --hiddenDimSize=[271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [271, 271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=100 --hiddenDimSize=[271,271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=100 --hiddenDimSize=[271,271] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=100 --hiddenDimSize=[271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [271, 271, 271]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=100 --hiddenDimSize=[271,271,271]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=100 --hiddenDimSize=[271,271,271] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=100 --hiddenDimSize=[271, 271, 271]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=100 --hiddenDimSize=[542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=100 --hiddenDimSize=[542] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=100 --hiddenDimSize=[542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [542, 542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=100 --hiddenDimSize=[542,542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=100 --hiddenDimSize=[542,542] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=100 --hiddenDimSize=[542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [542, 542, 542]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=100 --hiddenDimSize=[542,542,542]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=100 --hiddenDimSize=[542,542,542] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=100 --hiddenDimSize=[542, 542, 542]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1084]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=100 --hiddenDimSize=[1084]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=100 --hiddenDimSize=[1084] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=100 --hiddenDimSize=[1084]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1084, 1084]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=100 --hiddenDimSize=[1084,1084]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=100 --hiddenDimSize=[1084,1084] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=100 --hiddenDimSize=[1084, 1084]")
      print(stdout.decode().split("\n"))
      print("\n")

  # cell [1084, 1084, 1084]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855" compiled_models/855-encdec-model --attentionDimSize=100 --hiddenDimSize=[1084,1084,1084]', shell=True)
  best_model = stdout.decode().split("\n")[-5].split(" ")[-1]
  stdout = subprocess.check_output('python3.7 src/tf/LSTM-tf-Google-test.py "../clinical-prediction/data/mimic-iii/icd9/mimic_90-10_855"' + " \"" + best_model + "\"" + ' --attentionDimSize=100 --hiddenDimSize=[1084,1084,1084] --runName=Att_c-MIMIC-855_', shell=True)

  with open(str(r) + '_encdec-mimic-855-experiments.txt', 'a') as f:
    with redirect_stdout(f):
      print("--attentionDimSize=100 --hiddenDimSize=[1084, 1084, 1084]")
      print(stdout.decode().split("\n"))
      print("\n")