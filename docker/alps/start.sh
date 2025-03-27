# build
srun -p debug --cpus-per-task 32 --mem 64G -N 1 --gpus 4 --pty --account a-g34 bash
srun --reservation interactive_jobs --cpus-per-task 32 --mem 64G -N 1 --gpus 4 --pty --account a-g34 bash
# env, cpu-only
srun -p debug --cpus-per-task 32 --mem 64G -N 1 --gpus 4 --environment=/users/ljiayong/projects/DCVC/docker/alps/clariden.toml --pty --account a-g34 bash
srun --reservation interactive_jobs --cpus-per-task 32 --mem 64G -N 1 --gpus 4 --environment=/users/ljiayong/projects/DCVC/docker/alps/clariden.toml --pty --account a-g34 bash