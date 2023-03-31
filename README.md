# eos
Empirical Analysis of Edge of Stability


Usage
python gd.py synthetic-cifar-1 fc-tanh-width8-no-bias mse 10.0 5000 --loss_goal 0.00 --eig_freq 100 --save_freq 100 --seed 3 --neigs 10 --eos_log -1 --param_save -1 --grad_step -1 --iterate_freq -1

python gd.py cifar10-1k nfc-tanh-width8-no-bias mse 0.005 50000 --loss_goal 0.00 --eig_freq 500 --save_freq 500 --seed 3 --neigs 10 --eos_log -1 --param_save -1 --grad_step -1 --iterate_freq -1