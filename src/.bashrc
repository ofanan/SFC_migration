# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
conda init bash
conda activate py37
export SUMO_HOME="/home/itamarq/.conda/envs/py37/lib/python3.7/site-packages/sumo/"
export PYTHON_PATH="/home/itamarq/.conda/envs/py37/lib/python3.7/site-packages/sumolib"
source .myalias
cd itamarq/SFC_migration/src


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/storage/modules/packages/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/storage/modules/packages/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/storage/modules/packages/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/storage/modules/packages/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

