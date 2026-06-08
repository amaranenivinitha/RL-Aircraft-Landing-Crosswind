import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AircraftLandingEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        wind_intensity=1.0,
        turbulence_intensity=1.0
    ):

        super().__init__()

        self.wind_intensity = wind_intensity
        self.turbulence_intensity = turbulence_intensity

        high = np.array([
            1000.0,   # altitude
            10.0,     # vertical speed
            1.0,      # pitch
            1.0,      # throttle
            100.0,    # lateral position
            10.0,     # lateral speed
            10.0,     # wind estimate
            1000.0    # runway distance
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=np.array([-1, 0, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        self.max_steps = 600

        self.reset()

    # =====================================================
    # RESET
    # =====================================================

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.altitude = 400.0
        self.vertical_speed = -2.0

        self.pitch = 0.0
        self.throttle = 0.5

        self.lateral_pos = np.random.uniform(-5, 5)
        self.lateral_speed = 0.0

        self.wind_estimate = 0.0

        self.steps = 0

        return self._get_obs(), {}

    # =====================================================
    # OBSERVATION
    # =====================================================

    def _get_obs(self):

        runway_distance = self.altitude

        return np.array([
            self.altitude,
            self.vertical_speed,
            self.pitch,
            self.throttle,
            self.lateral_pos,
            self.lateral_speed,
            self.wind_estimate,
            runway_distance
        ], dtype=np.float32)

    # =====================================================
    # STEP
    # =====================================================

    def step(self, action):

        pitch_cmd, throttle_cmd, roll_cmd, rudder_cmd = action

        # -------------------------------------------------
        # THROTTLE
        # -------------------------------------------------

        self.throttle += 0.02 * throttle_cmd

        self.throttle = np.clip(
            self.throttle,
            0.0,
            1.0
        )

        # -------------------------------------------------
        # PITCH
        # -------------------------------------------------

        self.pitch += 0.02 * pitch_cmd

        self.pitch = np.clip(
            self.pitch,
            -0.5,
            0.5
        )

        # -------------------------------------------------
        # VERTICAL DYNAMICS
        # -------------------------------------------------

        gravity = -0.06

        lift = self.pitch * 0.04

        thrust = self.throttle * 0.05

        self.vertical_speed += (
            gravity
            + lift
            + thrust
        )

        self.vertical_speed = np.clip(
            self.vertical_speed,
            -5.0,
            2.0
        )

        self.altitude += self.vertical_speed

        # -------------------------------------------------
        # CROSSWIND + TURBULENCE
        # -------------------------------------------------

        wind_force = np.random.normal(
            0,
            self.wind_intensity * 0.15
        )

        turbulence = np.random.normal(
            0,
            self.turbulence_intensity * 0.05
        )

        self.wind_estimate = wind_force

        self.lateral_speed += (
            roll_cmd * 0.6
            + rudder_cmd * 0.2
            + wind_force
            + turbulence
            - 0.08 * self.lateral_speed
        )

        self.lateral_speed = np.clip(
            self.lateral_speed,
            -5.0,
            5.0
        )

        self.lateral_pos += self.lateral_speed

        # -------------------------------------------------
        # REWARD FUNCTION
        # -------------------------------------------------

        reward = 0.0

        # stay close to centerline
        reward -= 0.2 * abs(self.lateral_pos)

        # maintain target descent
        reward -= 0.5 * abs(
            self.vertical_speed + 1.5
        )

        # smooth controls
        reward -= 0.05 * abs(pitch_cmd)

        reward -= 0.05 * abs(roll_cmd)

        # progress toward runway
        reward += 0.03 * (
            400.0 - self.altitude
        )

        # flare reward
        if self.altitude < 20:

            reward += (
                max(
                    0.0,
                    2.0 - abs(self.vertical_speed)
                )
                * 5.0
            )

        terminated = False
        truncated = False

        success = False
        landing_type = "Flying"

        # -------------------------------------------------
        # TOUCHDOWN
        # -------------------------------------------------

        if self.altitude <= 0:

            terminated = True

            self.altitude = 0

            if (
                abs(self.vertical_speed) <= 2.0
                and abs(self.lateral_pos) <= 5.0
            ):

                reward += 10000

                success = True

                landing_type = "Smooth Landing"

            elif (
                abs(self.vertical_speed) <= 3.0
                and abs(self.lateral_pos) <= 10.0
            ):

                reward += 5000

                success = True

                landing_type = "Hard Landing"

            else:

                reward -= 5000

                success = False

                landing_type = "Crash"

        self.steps += 1

        if self.steps >= self.max_steps:

            truncated = True

        info = {
            "success": success,
            "landing_type": landing_type
        }

        return (
            self._get_obs(),
            float(reward),
            terminated,
            truncated,
            info
        )