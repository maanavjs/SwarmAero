#!/usr/bin/env python3
"""
SwarmAero Quaternion EKF
State vector: [pos(3), vel(3), quat(4), accel_bias(3), gyro_bias(3)] = 16 states
Author: Maanav Shah
"""

import numpy as np
from scipy.spatial.transform import Rotation


class QuaternionEKF:
    def __init__(self):
        # ── State ──────────────────────────────────────────────────────────
        # x = [px, py, pz, vx, vy, vz, qw, qx, qy, qz, bax, bay, baz, bgx, bgy, bgz]
        self.x = np.zeros(16)
        self.x[6] = 1.0  # qw = 1 (identity quaternion)

        # ── Covariance ─────────────────────────────────────────────────────
        self.P = np.eye(16) * 0.1
        self.P[6:10, 6:10] = np.eye(4) * 0.01

        # ── Process noise Q ────────────────────────────────────────────────
        self.Q = np.zeros((16, 16))
        self.Q[0:3,   0:3]   = np.eye(3) * 1e-4  # position
        self.Q[3:6,   3:6]   = np.eye(3) * 1e-3  # velocity
        self.Q[6:10,  6:10]  = np.eye(4) * 1e-5  # quaternion
        self.Q[10:13, 10:13] = np.eye(3) * 1e-6  # accel bias random walk
        self.Q[13:16, 13:16] = np.eye(3) * 1e-7  # gyro bias random walk

        # ── Measurement noise R (GPS: pos + vel) ───────────────────────────
        self.R_gps = np.diag([0.5**2, 0.5**2, 0.8**2,
                               0.1**2, 0.1**2, 0.1**2])

        # ── Constants ──────────────────────────────────────────────────────
        self.g = np.array([0.0, 0.0, -9.81])

        # ── NIS/NEES logging ───────────────────────────────────────────────
        self.nis_log  = []
        self.nees_log = []
        self.time_log = []

    # ═══════════════════════════════════════════════════════════════════════
    # HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def quat(self):
        """Return normalized quaternion [qw, qx, qy, qz]."""
        q = self.x[6:10]
        return q / np.linalg.norm(q)

    def rot_matrix(self):
        """Rotation matrix body → world frame."""
        q = self.quat()
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()

    @staticmethod
    def omega_matrix(w):
        """4x4 Omega matrix for quaternion kinematics: q_dot = 0.5 * Omega(w) * q"""
        wx, wy, wz = w
        return np.array([
            [ 0,  -wx, -wy, -wz],
            [ wx,   0,  wz, -wy],
            [ wy, -wz,   0,  wx],
            [ wz,  wy, -wx,   0]
        ])

    @staticmethod
    def skew(v):
        """3x3 skew-symmetric matrix."""
        return np.array([
            [ 0,    -v[2],  v[1]],
            [ v[2],  0,    -v[0]],
            [-v[1],  v[0],  0   ]
        ])

    # ═══════════════════════════════════════════════════════════════════════
    # PREDICTION STEP
    # ═══════════════════════════════════════════════════════════════════════

    def predict(self, accel_meas, gyro_meas, dt):
        """
        Propagate state using IMU measurements.
        accel_meas : [ax, ay, az] body frame (m/s^2)
        gyro_meas  : [wx, wy, wz] body frame (rad/s)
        dt         : timestep (s)
        """
        accel_bias = self.x[10:13]
        gyro_bias  = self.x[13:16]
        a_body = accel_meas - accel_bias
        w_body = gyro_meas  - gyro_bias

        pos = self.x[0:3].copy()
        vel = self.x[3:6].copy()
        q   = self.quat()

        # Rotate accel to world frame and add gravity
        R       = self.rot_matrix()
        a_world = R @ a_body + self.g

        # Integrate position and velocity
        pos_new = pos + vel * dt + 0.5 * a_world * dt**2
        vel_new = vel + a_world * dt

        # Exact discrete quaternion update (Rodrigues rotation)
        angle = np.linalg.norm(w_body) * dt
        if angle > 1e-10:
            axis = w_body / np.linalg.norm(w_body)
            dq = np.array([
                np.cos(angle / 2),
                axis[0] * np.sin(angle / 2),
                axis[1] * np.sin(angle / 2),
                axis[2] * np.sin(angle / 2)
            ])
        else:
            dq = np.array([1.0,
                           0.5 * w_body[0] * dt,
                           0.5 * w_body[1] * dt,
                           0.5 * w_body[2] * dt])

        # q_new = q ⊗ dq
        qw, qx, qy, qz = q
        dw, dx, dy, dz = dq
        q_new = np.array([
            qw*dw - qx*dx - qy*dy - qz*dz,
            qw*dx + qx*dw + qy*dz - qz*dy,
            qw*dy - qx*dz + qy*dw + qz*dx,
            qw*dz + qx*dy - qy*dx + qz*dw
        ])
        q_new /= np.linalg.norm(q_new)  # re-normalize every step

        self.x = np.concatenate([pos_new, vel_new, q_new, accel_bias, gyro_bias])

        # ── Jacobian F (16x16) ─────────────────────────────────────────────
        F = np.eye(16)
        F[0:3, 3:6]   = np.eye(3) * dt
        F[3:6, 6:10]  = self._dRa_dq(q, a_body) * dt
        F[3:6, 10:13] = -R * dt
        Omega = self.omega_matrix(w_body)
        F[6:10, 6:10]  = np.eye(4) + 0.5 * Omega * dt
        F[6:10, 13:16] = -0.5 * self._dOmega_dbias(q) * dt

        # Covariance propagation: P = FPF^T + Q
        self.P = F @ self.P @ F.T + self.Q

    def _dRa_dq(self, q, a):
        """Jacobian of R(q)*a w.r.t. quaternion. Shape: (3, 4)"""
        qw, qx, qy, qz = q
        ax, ay, az = a
        dRa_dqw = 2 * np.array([
             qw*ax - qz*ay + qy*az,
             qz*ax + qw*ay - qx*az,
            -qy*ax + qx*ay + qw*az])
        dRa_dqx = 2 * np.array([
             qx*ax + qy*ay + qz*az,
             qy*ax - qx*ay - qw*az,
             qz*ax + qw*ay - qx*az])  # corrected sign
        dRa_dqy = 2 * np.array([
            -qy*ax + qx*ay + qw*az,
             qx*ax + qy*ay + qz*az,
            -qw*ax + qz*ay - qy*az])  # corrected sign
        dRa_dqz = 2 * np.array([
            -qz*ax - qw*ay + qy*az,
             qw*ax - qz*ay + qy*az,
             qx*ax + qy*ay + qz*az])  # corrected sign
        return np.column_stack([dRa_dqw, dRa_dqx, dRa_dqy, dRa_dqz])

    def _dOmega_dbias(self, q):
        """
        Jacobian of (0.5 * Omega(w)*q) w.r.t. gyro bias. Shape: (4, 3)
        Negative because bias is subtracted from measurement.
        """
        qw, qx, qy, qz = q
        return np.array([
            [ qx,  qy,  qz],
            [-qw,  qz, -qy],
            [-qz, -qw,  qx],
            [ qy, -qx, -qw]
        ])

    # ═══════════════════════════════════════════════════════════════════════
    # GPS UPDATE STEP
    # ═══════════════════════════════════════════════════════════════════════

    def update_gps(self, gps_pos, gps_vel, timestamp=None, ground_truth=None):
        """
        GPS measurement update using position + velocity.
        gps_pos : [px, py, pz] world frame (m)
        gps_vel : [vx, vy, vz] world frame (m/s)
        """
        # Measurement matrix H: maps state to [pos, vel]
        H = np.zeros((6, 16))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)

        z      = np.concatenate([gps_pos, gps_vel])
        z_pred = H @ self.x
        nu     = z - z_pred  # innovation

        # Innovation covariance
        S = H @ self.P @ H.T + self.R_gps

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ nu

        # Re-normalize quaternion after update
        self.x[6:10] /= np.linalg.norm(self.x[6:10])

        # Joseph form covariance update — numerically stable, guarantees PSD
        I_KH = np.eye(16) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_gps @ K.T

        # NIS: ε = ν^T S^{-1} ν — should be chi²(6)
        nis = float(nu.T @ np.linalg.inv(S) @ nu)
        self.nis_log.append(nis)

        # NEES: ε = ẽ^T P^{-1} ẽ — requires ground truth from Gazebo
        if ground_truth is not None:
            e    = ground_truth[0:6] - self.x[0:6]
            P_pv = self.P[0:6, 0:6]
            nees = float(e.T @ np.linalg.inv(P_pv) @ e)
            self.nees_log.append(nees)

        if timestamp is not None:
            self.time_log.append(timestamp)

        return nu, S, nis

    # ═══════════════════════════════════════════════════════════════════════
    # NIS/NEES PLOTTING
    # ═══════════════════════════════════════════════════════════════════════

    def plot_consistency(self, save_path='results/nis_nees.png'):
        """Plot NIS and NEES with 95% chi-squared bounds."""
        import matplotlib.pyplot as plt
        from scipy.stats import chi2

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        t = np.arange(len(self.nis_log))

        # NIS — chi²(6) bounds (6 = GPS measurement dimension)
        dof_nis = 6
        lower_nis = chi2.ppf(0.025, dof_nis)
        upper_nis = chi2.ppf(0.975, dof_nis)

        axes[0].plot(t, self.nis_log, 'b-', linewidth=0.8, label='NIS')
        axes[0].axhline(upper_nis, color='r', linestyle='--',
                        label=f'95% upper bound ({upper_nis:.2f})')
        axes[0].axhline(lower_nis, color='r', linestyle='--',
                        label=f'95% lower bound ({lower_nis:.2f})')
        axes[0].set_title('Normalized Innovation Squared (NIS) — GPS Update')
        axes[0].set_xlabel('Update step')
        axes[0].set_ylabel('NIS value')
        axes[0].legend()
        axes[0].grid(True)

        # NEES — chi²(6) bounds (pos+vel = 6 states)
        if self.nees_log:
            dof_nees = 6
            lower_nees = chi2.ppf(0.025, dof_nees)
            upper_nees = chi2.ppf(0.975, dof_nees)
            t2 = np.arange(len(self.nees_log))

            axes[1].plot(t2, self.nees_log, 'g-', linewidth=0.8, label='NEES')
            axes[1].axhline(upper_nees, color='r', linestyle='--',
                            label=f'95% upper bound ({upper_nees:.2f})')
            axes[1].axhline(lower_nees, color='r', linestyle='--',
                            label=f'95% lower bound ({lower_nees:.2f})')
            axes[1].set_title('Normalized Estimation Error Squared (NEES) — vs Gazebo Ground Truth')
            axes[1].set_xlabel('Update step')
            axes[1].set_ylabel('NEES value')
            axes[1].legend()
            axes[1].grid(True)
        else:
            axes[1].text(0.5, 0.5, 'No NEES data (ground truth not provided)',
                        ha='center', va='center', transform=axes[1].transAxes)

        plt.tight_layout()
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f'NIS/NEES plot saved to {save_path}')