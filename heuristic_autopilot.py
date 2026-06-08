import numpy as np

from aircraft_env import AircraftLandingEnv


# =====================================================
# HEURISTIC CONTROLLER
# =====================================================

def heuristic_policy(obs):

    alt, vs, pitch, throttle, lat_pos, lat_speed, wind_est, runway_dist = obs

    # -------------------------------------------------
    # TARGET DESCENT PROFILE
    # -------------------------------------------------

    if alt > 120:

        target_vs = -2.0

    elif alt > 60:

        target_vs = -1.2

    elif alt > 25:

        target_vs = -0.6

    elif alt > 10:

        target_vs = -0.25

    else:

        target_vs = -0.05

    # -------------------------------------------------
    # THROTTLE CONTROL
    # -------------------------------------------------

    throttle_cmd = 0.6 + 0.25 * (target_vs - vs)

    throttle_cmd = np.clip(
        throttle_cmd,
        0.0,
        1.0
    )

    # -------------------------------------------------
    # PITCH CONTROL
    # -------------------------------------------------

    pitch_cmd = 0.2 * (target_vs - vs)

    pitch_cmd = np.clip(
        pitch_cmd,
        -1.0,
        1.0
    )

    # -------------------------------------------------
    # LATERAL CONTROL
    # -------------------------------------------------

    kp_lat = 0.08

    kd_lat = 0.45

    roll_cmd = -(
        kp_lat * lat_pos
        + kd_lat * lat_speed
    )

    roll_cmd = np.clip(
        roll_cmd,
        -1.0,
        1.0
    )

    rudder_cmd = -0.2 * lat_speed

    rudder_cmd = np.clip(
        rudder_cmd,
        -1.0,
        1.0
    )

    return np.array([
        pitch_cmd,
        throttle_cmd,
        roll_cmd,
        rudder_cmd
    ], dtype=np.float32)


# =====================================================
# EVALUATION
# =====================================================

episodes = 100

success_count = 0

smooth_count = 0

hard_count = 0

crash_count = 0

total_rewards = []

print("\n=== HEURISTIC CONTROLLER EVALUATION ===\n")

for ep in range(episodes):

    env = AircraftLandingEnv(

        wind_intensity=np.random.uniform(0.5, 3.0),

        turbulence_intensity=np.random.uniform(0.5, 3.0)

    )

    obs, _ = env.reset()

    done = False

    total_reward = 0

    final_info = {}

    while not done:

        action = heuristic_policy(obs)

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        total_reward += reward

        final_info = info

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

    total_rewards.append(total_reward)

    print(
        f"Episode {ep+1:03d} | "
        f"Reward: {total_reward:.2f} | "
        f"{landing_type}"
    )


print("\n========== FINAL RESULTS ==========")

print(f"Total Episodes: {episodes}")

print(
    f"Success Rate: "
    f"{(success_count / episodes) * 100:.2f}%"
)

print(f"Average Reward: {np.mean(total_rewards):.2f}")

print(f"Smooth Landings: {smooth_count}")

print(f"Hard Landings: {hard_count}")

print(f"Crashes: {crash_count}")