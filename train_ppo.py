import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import PPO

from aircraft_env import AircraftLandingEnv


print("🚀 Training PPO Agent for Autonomous Aircraft Landing")


# =====================================================
# TRAINING ENVIRONMENT
# =====================================================

env = gym.wrappers.TimeLimit(
    AircraftLandingEnv(
        wind_intensity=1.5,
        turbulence_intensity=1.5
    ),
    max_episode_steps=600
)


# =====================================================
# PPO MODEL
# =====================================================

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    batch_size=256,
    n_steps=4096,
    gamma=0.995,
    gae_lambda=0.98,
    clip_range=0.15,
    ent_coef=0.005,
    tensorboard_log="./ppo_tensorboard/"
)


# =====================================================
# TRAINING SETTINGS
# =====================================================

TOTAL_TIMESTEPS = 500000
EVAL_INTERVAL = 25000

reward_history = []
success_history = []
timestep_history = []

print(
    f"\nStarting training for "
    f"{TOTAL_TIMESTEPS} timesteps...\n"
)


# =====================================================
# EVALUATION FUNCTION
# =====================================================

def evaluate_model(model, episodes=20):

    successes = 0
    rewards = []

    for _ in range(episodes):

        test_env = AircraftLandingEnv(
            wind_intensity=np.random.uniform(0.5, 3.0),
            turbulence_intensity=np.random.uniform(0.5, 3.0)
        )

        obs, _ = test_env.reset()

        done = False

        total_reward = 0

        final_info = {}

        while not done:

            action, _ = model.predict(
                obs,
                deterministic=True
            )

            obs, reward, terminated, truncated, info = (
                test_env.step(action)
            )

            done = terminated or truncated

            total_reward += reward

            final_info = info

        rewards.append(total_reward)

        if final_info.get("success", False):
            successes += 1

    return (
        np.mean(rewards),
        100.0 * successes / episodes
    )


# =====================================================
# TRAINING LOOP
# =====================================================

current_steps = 0

while current_steps < TOTAL_TIMESTEPS:

    model.learn(
        total_timesteps=EVAL_INTERVAL,
        reset_num_timesteps=False
    )

    current_steps += EVAL_INTERVAL

    avg_reward, success_rate = evaluate_model(
        model,
        episodes=20
    )

    reward_history.append(avg_reward)

    success_history.append(success_rate)

    timestep_history.append(current_steps)

    print(
        f"\nTimesteps: {current_steps}"
    )

    print(
        f"Average Reward: {avg_reward:.2f}"
    )

    print(
        f"Success Rate: {success_rate:.1f}%"
    )

    model.save(
        f"checkpoint_{current_steps}"
    )


# =====================================================
# SAVE FINAL MODEL
# =====================================================

model.save("aircraft_landing_ppo")

print("\n✅ FINAL MODEL SAVED")


# =====================================================
# REWARD CURVE
# =====================================================

plt.figure(figsize=(10, 5))

plt.plot(
    timestep_history,
    reward_history,
    marker="o"
)

plt.title(
    "Average Reward vs Training Timesteps"
)

plt.xlabel("Timesteps")

plt.ylabel("Average Reward")

plt.grid(True)

plt.tight_layout()

plt.savefig("reward_curve.png")

plt.close()


# =====================================================
# SUCCESS RATE CURVE
# =====================================================

plt.figure(figsize=(10, 5))

plt.plot(
    timestep_history,
    success_history,
    marker="o"
)

plt.title(
    "Landing Success Rate vs Training Timesteps"
)

plt.xlabel("Timesteps")

plt.ylabel("Success Rate (%)")

plt.grid(True)

plt.tight_layout()

plt.savefig("success_rate_curve.png")

plt.close()


print("✅ reward_curve.png saved")

print("✅ success_rate_curve.png saved")