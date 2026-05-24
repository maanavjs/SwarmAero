#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
from px4_msgs.msg import SensorCombined, SensorGps, VehicleOdometry
from geometry_msgs.msg import PoseStamped
from swarm_ekf.quaternion_ekf import QuaternionEKF

QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

class EKFNode(Node):
    def __init__(self):
        super().__init__('swarm_ekf_node')
        self.ekf = QuaternionEKF()
        self.last_imu_time = None
        self.ground_truth  = None

        self.create_subscription(
            SensorCombined,
            '/fmu/out/sensor_combined',
            self.imu_callback, QOS)

        self.create_subscription(
            SensorGps,
            '/fmu/out/vehicle_gps_position',
            self.gps_callback, QOS)

        self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.truth_callback, QOS)

        self.est_pub = self.create_publisher(
            PoseStamped, '/swarm/ekf_estimate', 10)

        self.get_logger().info('SwarmAero EKF node started')

    def imu_callback(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_imu_time is None:
            self.last_imu_time = now
            return
        dt = now - self.last_imu_time
        self.last_imu_time = now
        if dt <= 0 or dt > 0.1:
            return
        accel = np.array([msg.accelerometer_m_s2[0],
                          msg.accelerometer_m_s2[1],
                          msg.accelerometer_m_s2[2]])
        gyro  = np.array([msg.gyro_rad[0],
                          msg.gyro_rad[1],
                          msg.gyro_rad[2]])
        self.ekf.predict(accel, gyro, dt)
        self._publish_estimate()

    def gps_callback(self, msg):
        gps_pos = np.array([float(msg.latitude_deg),
                             float(msg.longitude_deg),
                             float(msg.altitude_msl_m)])
        gps_vel = np.array([float(msg.vel_n_m_s),
                             float(msg.vel_e_m_s),
                             float(msg.vel_d_m_s)])
        t = self.get_clock().now().nanoseconds * 1e-9
        self.ekf.update_gps(gps_pos, gps_vel,
                            timestamp=t,
                            ground_truth=self.ground_truth)

    def truth_callback(self, msg):
        self.ground_truth = np.array([
            msg.position[0], msg.position[1], msg.position[2],
            msg.velocity[0], msg.velocity[1], msg.velocity[2],
            msg.q[0], msg.q[1], msg.q[2], msg.q[3],
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ])

    def _publish_estimate(self):
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        x = self.ekf.x
        msg.pose.position.x = float(x[0])
        msg.pose.position.y = float(x[1])
        msg.pose.position.z = float(x[2])
        q = self.ekf.quat()
        msg.pose.orientation.w = float(q[0])
        msg.pose.orientation.x = float(q[1])
        msg.pose.orientation.y = float(q[2])
        msg.pose.orientation.z = float(q[3])
        self.est_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Saving NIS/NEES plots...')
        node.ekf.plot_consistency(
            save_path='/home/maanavjugalshah/SwarmAero/results/nis_nees.png')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
