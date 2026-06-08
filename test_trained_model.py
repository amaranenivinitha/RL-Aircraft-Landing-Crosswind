import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from aircraft_env import AircraftLandingEnv


def classify_landing(info):

    return info["landing_type"], info["success"]


model = PPO.load("aircraft_landing_ppo")

scenarios = [
    ("Calm", 0.5),
    ("Normal", 1.0),
    ("Stormy", 2.0)
]

print("\n=== Aircraft Landing Test Results ===\n")

for name, wind in scenarios:

    env = AircraftLandingEnv(
        wind_intensity=wind,
        turbulence_intensity=wind
    )

    obs, _ = env.reset()

    done = False

    total_reward = 0

    altitude_trace = []

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, done, _, info = env.step(action)

        total_reward += reward

        altitude_trace.append(env.altitude)

    landing_type, success = classify_landing(info)

    print(
        f"{name} | "
        f"Reward: {total_reward:.2f} | "
        f"Landing: {landing_type} | "
        f"Success: {success}"
    )

    plt.figure()

    plt.plot(altitude_trace)

    plt.title(f"Altitude Trajectory - {name}")

    plt.xlabel("Step")

    plt.ylabel("Altitude (m)")

    plt.grid(True)

    plt.savefig(f"landing_trajectory_{name}.png")

    plt.close()