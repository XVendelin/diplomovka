"""
Tracked robot dynamics simulation.
Python equivalent of MATLAB's trackedRobotDynamics function.
"""
import numpy as np


def tracked_robot_dynamics(state, M, dt):
    """
    Dynamic model of a 4-tracked robot.

    Parameters:
    -----------
    state : np.ndarray, shape (7,)
        [x, y, theta, v, omega, z, zdot]
        x, y: position [m]
        theta: heading angle [rad]
        v: linear velocity [m/s]
        omega: angular velocity [rad/s]
        z: height [m]
        zdot: vertical velocity [m/s]

    M : np.ndarray, shape (4,)
        Motor torques [N*m]: [M_FL, M_FR, M_RL, M_RR]
        FL: Front Left, FR: Front Right, RL: Rear Left, RR: Rear Right

    dt : float
        Time step [s]

    Returns:
    --------
    state_new : np.ndarray, shape (7,)
        Updated state after dt seconds
    """
    # === ROBOT PARAMETERS (default values) ===
    m = 40.0  # [kg] robot mass
    Jz = 3.0  # [kg*m^2] moment of inertia around vertical axis
    y_offset = 0.25  # [m] half of track width
    r_s = 0.06  # [m] drive wheel radius
    mu = 0.7  # [-] friction coefficient
    eta_drive = 0.9  # [-] drivetrain efficiency
    b_v = 6.0  # [N/(m/s)] linear drag
    b_omega = 0.2  # [N*m*s/rad] rotational damping
    g = 9.80665  # [m/s^2] gravity

    # Active suspension parameters
    k_s = 10000.0  # [N/m] spring stiffness
    c_s = 200.0  # [N*s/m] damping
    z_eq = 0.20  # [m] equilibrium height
    z_cmd = 0.22  # [m] commanded height (active control)
    act_kp = 8000.0  # [N/m] proportional gain for height control
    act_kd = 300.0  # [N*s/m] derivative gain for height control

    # === UNPACK STATE ===
    x = state[0]
    y = state[1]
    theta = state[2]
    v = state[3]
    omega = state[4]
    z = state[5]
    zdot = state[6]

    # === CONVERT TORQUE TO TRACTION FORCE ===
    F_tang = (M / r_s) * eta_drive  # [N]

    N_total = m * g
    N_per_track = N_total / 4
    F_max = mu * N_per_track  # friction limit

    # Saturate forces
    F = np.clip(F_tang, -F_max, F_max)
    F_FL, F_FR, F_RL, F_RR = F

    F_L = F_FL + F_RL  # Left side total force
    F_R = F_FR + F_RR  # Right side total force

    # === LONGITUDINAL AND ROTATIONAL DYNAMICS ===
    F_x = F_L + F_R
    Mz = (F_R - F_L) * y_offset  # Yaw moment

    a_long = (F_x - b_v * v) / m
    a_yaw = (Mz - b_omega * omega) / Jz

    # === HEIGHT DYNAMICS (ACTIVE SUSPENSION) ===
    F_spring = -k_s * (z - z_eq)
    F_damp = -c_s * zdot
    F_active = act_kp * (z_cmd - z) - act_kd * zdot
    Fz_total = F_spring + F_damp + F_active - m * g
    zddot = Fz_total / m

    # === INTEGRATE VELOCITIES ===
    v_new = v + a_long * dt
    omega_new = omega + a_yaw * dt
    zdot_new = zdot + zddot * dt

    # === INTEGRATE POSITION ===
    if abs(omega_new) < 1e-6:
        # Straight line motion
        x_new = x + v_new * dt * np.cos(theta)
        y_new = y + v_new * dt * np.sin(theta)
        theta_new = theta
    else:
        # Circular motion
        theta_new = theta + omega_new * dt
        x_new = x + (v_new / omega_new) * (np.sin(theta_new) - np.sin(theta))
        y_new = y - (v_new / omega_new) * (np.cos(theta_new) - np.cos(theta))

    z_new = z + zdot_new * dt

    # Wrap angle to [-pi, pi]
    theta_new = ((theta_new + np.pi) % (2 * np.pi)) - np.pi

    # === ASSEMBLE NEW STATE ===
    state_new = np.array([x_new, y_new, theta_new, v_new, omega_new, z_new, zdot_new])

    return state_new