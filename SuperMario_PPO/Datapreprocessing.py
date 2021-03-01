from collections import deque

import cv2
import gym
import numpy as np

class EpisodicLifeEnv( gym.Wrapper ):
    def __init__(self, env):
        gym.Wrapper.__init__( self, env )
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step( action )
        if self.env.unwrapped._flag_get:
            reward += 100
            done = True
        if self.env.unwrapped._is_dying:
            reward -= 50
            done = True
        self.was_real_done = done
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset( **kwargs )
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step( 0 )
        return obs


class RewardScaler( gym.RewardWrapper ):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """

    def reward(self, reward):
        return reward * 0.05


class PreprocessFrame( gym.ObservationWrapper ):
    """
    Here we do the preprocessing part:
    - Set frame to gray
    - Resize the frame to 96x96x1
    """

    def __init__(self, env):
        gym.ObservationWrapper.__init__( self, env )
        self.width = 96
        self.height = 96
        self.observation_space = gym.spaces.Box( low=0, high=255,
                                                 shape=(self.height, self.width, 1), dtype=np.uint8 )

    def observation(self, frame):
        # Set frame to gray
        frame = cv2.cvtColor( frame, cv2.COLOR_RGB2GRAY )

        # Resize the frame to 96x96x1
        frame = cv2.resize( frame, (self.width, self.height), interpolation=cv2.INTER_AREA )
        frame = frame[:, :, None]

        return frame


class StochasticFrameSkip( gym.Wrapper ):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__( self, env )
        self.n = n
        self.stickprob = stickprob
        self.current_action = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr( env, "supports_want_render" )

    def reset(self, **kwargs):
        self.current_action = None
        return self.env.reset( **kwargs )

    def step(self, action):
        observation, info, done = None, None, False
        totry = 0
        for i in range( self.n ):
            # First step after reset, use action
            if self.current_action is None:
                self.current_action = action

            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.current_action = action
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.current_action = action

            if self.supports_want_render and i < self.n - 1:
                observation, rewind, done, info = self.env.step( self.current_action, want_render=False )
            else:
                observation, rewind, done, info = self.env.step( self.current_action )
            totry += rewind
            if done:
                break
        return observation, totry, done, info

    def seed(self, seed=None):
        self.rng.seed( seed )


class ScaledFloatFrame( gym.ObservationWrapper ):
    def __init__(self, env):
        gym.ObservationWrapper.__init__( self, env )
        self.observation_space = gym.spaces.Box( low=0, high=1, shape=env.observation_space.shape, dtype=np.float32 )

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array( observation ).astype( np.float32 ) / 255.0


class FrameStack( gym.Wrapper ):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__( self, env )
        self.k = k
        self.frames = deque( [], maxlen=k )
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box( low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)),
                                                 dtype=env.observation_space.dtype )

    def reset(self):
        ob = self.env.reset()
        for _ in range( self.k ):
            self.frames.append( ob )
        return self._get_observation()

    def step(self, action):
        ob, reward, done, info = self.env.step( action )
        self.frames.append( ob )
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        assert len( self.frames ) == self.k
        return LazyFrames( list( self.frames ) )


class LazyFrames( object ):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate( self._frames, axis=-1 )
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype( dtype )
        return out

    def __len__(self):
        return len( self._force() )

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]
