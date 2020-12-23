MODEL=$1
THRE=$2

python3 src/tf/LSTM-tf-Google-test.py "data/_855" $MODEL --strengthThreshold=$THRE