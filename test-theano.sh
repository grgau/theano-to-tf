MODEL=$1
THRE=$2

python src/theano/04.LSTM-Google-test.py "data/_855" $MODEL --strengthThreshold=$THRE
