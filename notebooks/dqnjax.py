import time
import jumanji
import flashbax as fbx

import jax
import flax
import optax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.training.train_state import TrainState
from jumanji.wrappers import AutoResetWrapper

from functools import partial

env = jumanji.make('Game2048-v1')
env = AutoResetWrapper(jumanji.make('Game2048-v1'))
# keys
key = jax.random.PRNGKey(1)
key, q_key, env_key = jax.random.split(key, 3)


#params

n_actions = env.action_spec.num_values
n_hidden1 = 64
n_hidden2 = 128
learning_rate = 1e-5
gamma = 0.95
batch_size = 64
buffer_size = 60000
min_buffer_size = 5000
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_steps = 20000
target_update_freq = 500
num_episodes = 1000
board_size = 4
num_envs = 16

class TeLU(nn.Module):
    @nn.compact
    def __call__(self, x):
      x = x.reshape((x.shape[0], -1))
      return x * nn.tanh(jnp.exp(x))

def preprocess_observation(obs):
    return jnp.log2(jnp.where(obs == 0, 1.0, obs.astype(jnp.float32)))


class QNetwork(nn.Module):
  n_actions: int
  n_hidden1: int
  n_hidden2: int

  @nn.compact
  def __call__(self, x):
    x = preprocess_observation(x)
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(self.n_hidden1)(x)
    x = TeLU()(x)
    x = nn.Dense(self.n_hidden2)(x)
    x = TeLU()(x)
    x = nn.Dense(self.n_actions)(x)
    return x

class Trainstate(TrainState):
  target_params: flax.core.FrozenDict

def create_replay_buffer(batch_size, max_length, min_length, board_size: int):
  q_learning_memory = fbx.make_item_buffer(
      max_length=max_length,
      min_length=min_length,
      sample_batch_size=batch_size,
      add_batches=True
  )

  dummy_transition = {
        "obs": jnp.zeros((board_size, board_size), dtype='float32'),
        "action": jnp.zeros((), dtype='int32'),
        "reward": jnp.zeros((), dtype='float32'),
        "next_obs": jnp.zeros((board_size, board_size), dtype='float32'),
        "action_mask": jnp.ones(n_actions, dtype='bool'),
        "next_action_mask": jnp.ones(n_actions, dtype='bool'),
        "done": jnp.zeros((), dtype='float32'),
    }

  q_learning_memory_state = q_learning_memory.init(dummy_transition)
  return q_learning_memory, q_learning_memory_state


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
  slope = (end_e - start_e) / duration
  return max(slope * t + start_e, end_e)

q_network = QNetwork(n_actions, n_hidden1=n_hidden1, n_hidden2=n_hidden2)

dummy_obs = jnp.zeros((1, board_size, board_size))

q_state = Trainstate.create(
    apply_fn = q_network.apply,
    params = q_network.init(q_key, dummy_obs),
    target_params = q_network.init(q_key, dummy_obs),
    tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate)
))


@jax.jit
def update(q_state, observations, actions, next_observations, rewards, dones, next_action_masks, gamma: float):

    q_next_target = q_network.apply(q_state.target_params, next_observations)
    masked_q_next = jnp.where(next_action_masks, q_next_target, -jnp.inf)
    q_next_target_max = jnp.max(masked_q_next, axis=-1)
    q_target = rewards + gamma * q_next_target_max * (1 - dones)

    q_pred_all = q_network.apply(q_state.params, observations)
    q_pred = jnp.take_along_axis(q_pred_all, actions[:, None], axis=-1)[:, 0]

    loss = jnp.mean(jnp.square(q_pred - q_target))

    grads = jax.grad(lambda params: jnp.mean(jnp.square(
        jnp.take_along_axis(q_network.apply(params, observations),
                           actions[:, None], axis=-1)[:, 0] - q_target)))(q_state.params)

    q_state = q_state.apply_gradients(grads=grads)
    return loss, q_pred, q_state


@partial(jax.jit, static_argnums=(4,))
def select_actions_vectorized(q_state, observations, action_masks, key, epsilon):
    batch_size = observations.shape[0]

    q_values = q_network.apply(q_state.params, observations)
    masked_q_values = jnp.where(action_masks, q_values, -jnp.inf)
    greedy_actions = jnp.argmax(masked_q_values, axis=-1)

    key, subkey = jax.random.split(key)
    random_actions = jax.random.randint(subkey, (batch_size,), 0, n_actions)

    key, subkey = jax.random.split(key)
    explore_mask = jax.random.uniform(subkey, (batch_size,)) < epsilon

    actions = jnp.where(explore_mask, random_actions, greedy_actions)
    valid_random_actions = jax.vmap(lambda mask, action:
                                   jax.lax.cond(~mask[action],
                                               lambda: jnp.argmax(mask.astype(jnp.int32)),
                                               lambda: action))(action_masks, random_actions)

    actions = jnp.where(explore_mask, valid_random_actions, greedy_actions)

    return actions, key

@jax.jit
def reset_envs(key):
  keys = jax.random.split(key, num_envs)
  states, timesteps = jax.vmap(env.reset)(keys)
  return states, timesteps

@jax.jit
def step_envs(states, actions):
  states, timesteps = jax.vmap(env.step)(states, actions)
  return states, timesteps

replay_buffer, buffer_state = create_replay_buffer(
    batch_size = batch_size,
    max_length = buffer_size,
    min_length = min_buffer_size,
    board_size = board_size,
)

def train_agent_vectorized(num_episodes):
    global key, q_state, buffer_state

    total_timesteps = 0
    episode_rewards = []
    losses = []


    key, reset_key = jax.random.split(key)
    env_states, timesteps = reset_envs(reset_key)
    episode_reward_accum = np.zeros(num_envs)
    episodes_completed = 0

    while episodes_completed < num_episodes:

        observations = timesteps.observation.board
        action_masks = timesteps.observation.action_mask


        epsilon = linear_schedule(epsilon_start, epsilon_end, epsilon_decay_steps, total_timesteps)
        key, action_key = jax.random.split(key)
        actions, key = select_actions_vectorized(q_state, observations, action_masks, action_key, epsilon)


        env_states, next_timesteps = step_envs(env_states, actions)
        next_observations = next_timesteps.observation.board
        next_action_masks = next_timesteps.observation.action_mask
        rewards = next_timesteps.reward
        dones = next_timesteps.last()


        rewards_np = np.array(rewards)
        dones_np = np.array(dones)
        episode_reward_accum += rewards_np


        transitions = {
            "obs": observations.astype("float32"),
            "action": actions.astype(jnp.int32),
            "reward": rewards.astype(jnp.float32),
            "next_obs": next_observations.astype("float32"),
            "done": dones.astype(jnp.float32),
            "action_mask": action_masks.astype("bool"),
            "next_action_mask": next_action_masks.astype("bool"),
        }
        buffer_state = replay_buffer.add(buffer_state, transitions)


        buffer_current_size = min(buffer_state.current_index, buffer_size)
        if buffer_current_size >= min_buffer_size:
            key, sample_key = jax.random.split(key)
            batch = replay_buffer.sample(buffer_state, sample_key)
            batch_data = batch.experience

            loss, _, q_state = update(
                q_state,
                batch_data["obs"],
                batch_data["action"],
                batch_data["next_obs"],
                batch_data["reward"],
                batch_data["done"],
                batch_data["next_action_mask"],
                gamma
            )
            losses.append(float(loss))


        if total_timesteps % target_update_freq == 0:
            q_state = q_state.replace(
                target_params=optax.incremental_update(
                    q_state.params,
                    q_state.target_params,
                    1.0
                )
            )


        done_indices = np.where(dones_np)[0]
        if len(done_indices) > 0:
            for env_idx in done_indices:
                episode_rewards.append(float(episode_reward_accum[env_idx]))
                episode_reward_accum[env_idx] = 0.0
                episodes_completed += 1

                if episodes_completed % 10 == 0:
                    avg_reward = np.mean(episode_rewards[-10:])
                    avg_loss = np.mean(losses[-100:]) if len(losses) >= 100 else 0
                    print(f"Episode {episodes_completed}, Avg Reward: {avg_reward:.2f}, "
                          f"Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.3f}, "
                          f"Buffer: {buffer_current_size}/{buffer_size}")


            key, reset_key = jax.random.split(key)
            reset_keys = jax.random.split(reset_key, num_envs)


            def maybe_reset(done, state, timestep, reset_key):
                new_state, new_timestep = env.reset(reset_key)
                return jax.lax.cond(
                    done,
                    lambda: (new_state, new_timestep),
                    lambda: (state, timestep)
                )

            env_states, timesteps = jax.vmap(maybe_reset)(
                dones, env_states, timesteps, reset_keys
            )
        else:
            timesteps = next_timesteps

        total_timesteps += num_envs

    return q_state, episode_rewards, losses

if __name__ == "__main__":
    print(f"Starting vectorized training with {num_envs} parallel environments...")
    print(f"Total environment steps per iteration: {num_envs}")
    trained_q_state, rewards, losses = train_agent_vectorized(num_episodes)
    print("Training completed!")
    print(f"Final average reward: {np.mean(rewards[-100:]):.2f}")

