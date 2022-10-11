# CUDA_VISIBLE_DEVICES=3 python main.py --cuda --resume base_LSTM_e40.pt --epochs 20 --lr 5 --emsize 1500 --nhid 1500 --batch_size 40 --dropout 0.65 --tied --scale 0.4 --save dre_LSTM_0.4_e20.pt
# CUDA_VISIBLE_DEVICES=3 python main.py --cuda --resume base_LSTM_e40.pt --epochs 20 --lr 5 --emsize 1500 --nhid 1500 --batch_size 40 --dropout 0.65 --tied --scale 0.35 --save dre_LSTM_0.35_e20.pt
# CUDA_VISIBLE_DEVICES=3 python main.py --cuda --resume base_LSTM_e40.pt --epochs 20 --lr 5 --emsize 1500 --nhid 1500 --batch_size 40 --dropout 0.65 --tied --scale 0.3 --save dre_LSTM_0.3_e20.pt
# python main.py --cuda --resume base_LSTM_e40.pt --epochs 10 --lr 5 --emsize 1500 --nhid 1500 --batch_size 40 --dropout 0.65 --tied --scale 0.2 --save dre_LSTM_0.2_e10.pt
# python main.py --cuda --resume base_LSTM_e40.pt --epochs 10 --lr 5 --emsize 1500 --nhid 1500 --batch_size 40 --dropout 0.65 --tied --scale 0.15 --save dre_LSTM_0.15_e10.pt
# python main.py --cuda --resume base_LSTM_e40.pt --epochs 10 --lr 5 --emsize 1500 --nhid 1500 --batch_size 40 --dropout 0.65 --tied --scale 0.1 --save dre_LSTM_0.1_e10.pt

# python main.py --cuda -e --emsize 1500 --nhid 1500 --batch_size 40 --dropout 0.65 --tied --resume dre_LSTM_0.4_e20.pt --scale 0.4
# python main.py --cuda -e --emsize 1500 --nhid 1500 --batch_size 40 --dropout 0.65 --tied --resume dre_LSTM_0.35_e20.pt --scale 0.35
# python main.py --cuda -e --emsize 1500 --nhid 1500 --batch_size 40 --dropout 0.65 --tied --resume dre_LSTM_0.3_e20.pt --scale 0.3
# python main.py --cuda -e --emsize 1500 --nhid 1500 --batch_size 40 --dropout 0.65 --tied --resume dre_LSTM_0.2_e10.pt --scale 0.2
# python main.py --cuda -e --emsize 1500 --nhid 1500 --batch_size 40 --dropout 0.65 --tied --resume dre_LSTM_0.15_e10.pt --scale 0.15
# python main.py --cuda -e --emsize 1500 --nhid 1500 --batch_size 40 --dropout 0.65 --tied --resume dre_LSTM_0.1_e10.pt --scale 0.1

# For HPCA'23 in-storage processing
python main.py --cuda -e --emsize 1500 --nhid 1500 --batch_size 40 --dropout 0.65 --tied --resume dre_LSTM_e10.pt --scale 0.25
