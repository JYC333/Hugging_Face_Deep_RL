setup:
	./setup_unit3.sh

train:
	python ./rl-baselines3-zoo/train.py --algo dqn --env SpaceInvadersNoFrameskip-v4 -f ./rl-baselines3-zoo/logs/

evaluate:
	python ./rl-baselines3-zoo/enjoy.py --algo dqn --env SpaceInvadersNoFrameskip-v4 --no-render --n-timesteps 5000 --folder ./rl-baselines3-zoo/logs/

publish:
	cd rl-baselines3-zoo/ && \
	python -m rl_zoo3.push_to_hub --algo dqn --env SpaceInvadersNoFrameskip-v4 --repo-name dqn-SpaceInvadersNoFrameskip-v4 -orga JYC333 -f logs/