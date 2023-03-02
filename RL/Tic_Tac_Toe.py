import numpy as np
from pettingzoo.classic import tictactoe_v3 # pettingzoo에서 tictactoe_v3 환경을 가져옴
from stable_baselines3.dqn import DQN, MlpPolicy # DQN과 MlpPolicy를 가져옴
from pantheonrl.common.agents import OffPolicyAgent # OffPolicyAgent를 가져옴
from pantheonrl.envs.pettingzoo import PettingZooAECWrapper

from datetime import datetime
from argparse import ArgumentParser

time_stamp = datetime.now().strftime("%Y%m%d_%H%M")

parser = ArgumentParser() # 설명 추가
parser.add_argument("--model", type=str, default="") # 모델 경로
args = parser.parse_args() # 파싱(구문 분석) 실행

env = PettingZooAECWrapper(tictactoe_v3.env(render_mode=None))
print(env.n_players)

dqn_policy = {'learning_rate': 1e-4,
              'buffer_size': 100000,
              'learning_starts': 5000,
              'batch_size': 64,
              'tau': 0.99,
              'train_freq': (1, 'episode'),
              'gradient_steps': -1,
              'target_update_interval': 200,
              'policy_kwargs': dict(net_arch=[64, 64]),
              'exploration_fraction': 0.3,
              'exploration_initial_eps': 0.99,
              'exploration_final_eps': 0.01,
              'tensorboard_log': './logs'
              }

for i in range(env.n_players - 1):
    partner = OffPolicyAgent(
        DQN(MlpPolicy,
            env.getDummyEnv(i),
            verbose=1, **dqn_policy),
    )
    env.add_partner_agent(partner, player_num=i+1)

model = DQN(MlpPolicy, env, verbose=2, **dqn_policy)

if args.model == '':
    model = model.learn(total_timesteps=500000,
                        eval_env=env,
                        eval_freq=20,
                        eval_log_path=f'./logs/ttt_{time_stamp}',
                        tb_log_name=f'ttt_{time_stamp}',
                        log_interval=20,
                        n_eval_episodes=10)
    model.save(f'last_model.zip')
else:
    model = model.load(args.model)

env2 = tictactoe_v3.env(render_mode='human')


def play(env, ego):
    env.reset()
    env.render()
    while True:
        if env.agent_selection == 'player_1':
            action = ego.policy.predict(env.observe('player_1')['observation'])
        else:
            action = int(input('action: '))
        if isinstance(action, tuple):
            action = action[0]
        if env.observe(env.agent_selection)['action_mask'][action] == 0:
            action = env.observe(env.agent_selection)[
                'action_mask'].tolist().index(1)

        print(f'{env.agent_selection} action: {action}')
        env.step(action)
        obs, reward, done1, done2, info = env.last()
        env.render()

        if done1 or done2:
            print(env.rewards)
            return env.rewards


play(env2, model)

# def check(replay_buffer, bs=None):
#     if bs is None:
#         bs = replay_buffer.buffer_size
#     for i in range(bs):
#         print(replay_buffer.get_transition(i))
