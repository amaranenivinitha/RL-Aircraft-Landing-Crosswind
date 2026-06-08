import numpy as np

from stable_baselines3 import PPO

from aircraft_env import AircraftLandingEnv


# =====================================================
# LOAD TRAINED MODEL
# =====================================================

model = PPO.load("aircraft_landing_ppo")


# =====================================================
# SETTINGS
# =====================================================

episodes = 100

success_count = 0

smooth_count = 0

hard_count = 0

crash_count = 0

rewards = []

lateral_errors = []

touchdown_vspeeds = []


print("\n=== PPO EVALUATION ===\n")


# =====================================================
# RUN EPISODES
# =====================================================

for ep in range(episodes):

    env = AircraftLandingEnv(

        wind_intensity=np.random.uniform(4.0, 6.0),

        turbulence_intensity=np.random.uniform(4.0, 6.0)

    )

    obs, _ = env.reset()

    done = False

    total_reward = 0

    final_info = {}

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        total_reward += reward

        final_info = info

    rewards.append(total_reward)

    # ==========================================
    # TOUCHDOWN METRICS
    # ==========================================

    lateral_error = abs(env.lateral_pos)

    touchdown_vspeed = abs(env.vertical_speed)

    lateral_errors.append(lateral_error)

    touchdown_vspeeds.append(touchdown_vspeed)

    # ==========================================
    # LANDING TYPE
    # ==========================================

    landing_type = final_info["landing_type"]

    success = final_info["success"]

    if success:

        success_count += 1

    if landing_type == "Smooth Landing":

        smooth_count += 1

    elif landing_type == "Hard Landing":

        hard_count += 1

    else:

        crash_count += 1

    print(
        f"Episode {ep+1:03d} | "
        f"Reward={total_reward:.2f} | "
        f"Lateral Error={lateral_error:.2f} m | "
        f"VSpeed={touchdown_vspeed:.2f} m/s | "
        f"{landing_type}"
    )


# =====================================================
# FINAL STATISTICS
# =====================================================

print("\n========== FINAL RESULTS ==========\n")

print(f"Total Episodes: {episodes}")

print(
    f"Success Rate: "
    f"{100 * success_count / episodes:.2f}%"
)

print(f"Smooth Landings: {smooth_count}")

print(f"Hard Landings: {hard_count}")

print(f"Crashes: {crash_count}")

print()

print(
    f"Average Reward: "
    f"{np.mean(rewards):.2f}"
)

print()

print(
    f"Mean Lateral Error: "
    f"{np.mean(lateral_errors):.3f} m"
)

print(
    f"Std Lateral Error: "
    f"{np.std(lateral_errors):.3f} m"
)

print(
    f"Max Lateral Error: "
    f"{np.max(lateral_errors):.3f} m"
)

print()

print(
    f"Mean Touchdown VSpeed: "
    f"{np.mean(touchdown_vspeeds):.3f} m/s"
)

print(
    f"Max Touchdown VSpeed: "
    f"{np.max(touchdown_vspeeds):.3f} m/s"
)