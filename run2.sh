hostname

source ~/.bashrc
conda activate pytorch201


python3 main_ei.py --exp-index $1
# python3 main_traunstein.py --exp-index $1


