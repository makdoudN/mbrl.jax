from gym.envs.registration import register

from .pusher import PusherEnv
from .cartpole import CartpoleEnv
from .halfcheetah import HalfCheetahEnv
from .peg_insertion_sawyer import PegEnv
from .reacher_sawyer import Reacher7DOFEnv
from .point_mass import PointMassEnv


register(id="MBRLCartpole-v0", entry_point=CartpoleEnv, max_episode_steps=200)
register(id="MBRLPusher-v0", entry_point=PusherEnv, max_episode_steps=150)
register(id="MBRLPointMass-v0", entry_point=PointMassEnv, max_episode_steps=50)
register(id="MBRLHalfCheetah-v0", entry_point=HalfCheetahEnv, max_episode_steps=1000)
register(id="MBRLPeg-v0", entry_point=PegEnv, max_episode_steps=200)
register(id="MBRLReacher-v0", entry_point=Reacher7DOFEnv, max_episode_steps=200)
