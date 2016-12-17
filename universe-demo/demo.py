import universe
import gym

env = gym.make('flashgames.DuskDrive-v0')
env.configure(remotes="vnc://localhost:5900+15899")
#env.configure(remotes=1)
observation_n = env.reset()

while True:
    # your agent generates action_n at 60 frames per second
    action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]
    observation_n, reward_n, done_n, info = env.step(action_n)
    env.render()
