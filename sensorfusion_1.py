#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 07:12:54 2023

@author: ifeoluwaolawore
"""

import math
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def invSqrt(x):
    halfx = 0.5 * x
    y = x
    i = int.from_bytes(struct.pack('f', y), byteorder='little')
    i = 0x5f3759df - (i >> 1)
    y = struct.unpack('f', i.to_bytes(4, byteorder='little', signed=False))[0]
    y = y * (1.5 - (halfx * y * y))
    return y


class SensorFusion:
    def __init__(self, TWO_KP_DEF, TWO_KI_DEF):
        self.twoKp = TWO_KP_DEF  # 2 * proportional gain (Kp)
        self.twoKi = TWO_KI_DEF  # 2 * integral gain (Ki)
        self.integralFBx = 0.0
        self.integralFBy = 0.0
        self.integralFBz = 0.0  # integral error terms scaled by Ki

        # quaternion of sensor frame relative to auxiliary frame
        self.qw = 1.0
        self.qx = 0.0
        self.qy = 0.0
        self.qz = 0.0

        # Unit vector in the estimated gravity direction
        self.gravX = 0.0
        self.gravY = 0.0
        self.gravZ = 0.0
        self.M_PI_F = 3.1416

        self.baseZacc = 1.0
        self.isInit = True
        self.isCalibrated = False

    def sensfusion6UpdateQ(self, gx, gy, gz, ax, ay, az, dt):
        self.sensfusion6UpdateQImpl(gx, gy, gz, ax, ay, az, dt)
        self.gravX, self.gravY, self.gravZ = self.estimatedGravityDirection()

        if not self.isCalibrated:
            self.baseZacc = self.sensfusion6GetAccZ(ax, ay, az)
            self.isCalibrated = True

    def sensfusion6UpdateQImpl(self, gx, gy, gz, ax, ay, az, dt):
        recipNorm = 0.0
        halfvx, halfvy, halfvz = 0.0, 0.0, 0.0
        halfex, halfey, halfez = 0.0, 0.0, 0.0
        qa, qb, qc = 0.0, 0.0, 0.0

        # Convert gyroscope degrees/sec to radians/sec
        gx *= 0.0174533
        gy *= 0.0174533
        gz *= 0.0174533

        # Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalization)
        if not ((ax == 0.0) and (ay == 0.0) and (az == 0.0)):
            # Normalise accelerometer measurement
            recipNorm = invSqrt(ax * ax + ay * ay + az * az)
            ax *= recipNorm
            ay *= recipNorm
            az *= recipNorm

            # Estimated direction of gravity and vector perpendicular to magnetic flux
            halfvx = self.qx * self.qz - self.qw * self.qy
            halfvy = self.qw * self.qx + self.qy * self.qz
            halfvz = self.qw * self.qw - 0.5 + self.qz * self.qz

            # Error is sum of cross product between estimated and measured direction of gravity
            halfex = (ay * halfvz - az * halfvy)
            halfey = (az * halfvx - ax * halfvz)
            halfez = (ax * halfvy - ay * halfvx)

            # Compute and apply integral feedback if enabled
            if self.twoKi > 0.0:
                self.integralFBx += self.twoKi * halfex * dt  # integral error scaled by Ki
                self.integralFBy += self.twoKi * halfey * dt
                self.integralFBz += self.twoKi * halfez * dt
                gx += self.integralFBx  # apply integral feedback
                gy += self.integralFBy
                gz += self.integralFBz
            else:
                self.integralFBx = 0.0  # prevent integral windup
                self.integralFBy = 0.0
                self.integralFBz = 0.0

            # Apply proportional feedback
            gx += self.twoKp * halfex
            gy += self.twoKp * halfey
            gz += self.twoKp * halfez

        # Integrate rate of change of quaternion
        gx *= (0.5 * dt)  # pre-multiply common factors
        gy *= (0.5 * dt)
        gz *= (0.5 * dt)
        qa = self.qw
        qb = self.qx
        qc = self.qy
        self.qw += (-qb * gx - qc * gy - self.qz * gz)
        self.qx += (qa * gx + qc * gz - self.qz * gy)
        self.qy += (qa * gy - qb * gz + self.qz * gx)
        self.qz += (qa * gz + qb * gy - qc * gx)

        # Normalize quaternion
        recipNorm = invSqrt(self.qw * self.qw + self.qx * self.qx + self.qy * self.qy + self.qz * self.qz)
        self.qw *= recipNorm
        self.qx *= recipNorm
        self.qy *= recipNorm
        self.qz *= recipNorm

    def sensfusion6GetEulerRPY(self):
        """Return Euler Angles"""
        gx = self.gravX
        gy = self.gravY
        gz = self.gravZ

        if gx > 1:
            gx = 1
        if gx < -1:
            gx = -1

        roll_rad = -math.asin(gx)
        pitch_rad = math.atan2(gy, gz)
        yaw_rad = math.atan2(
            2 * (self.qw * self.qz + self.qx * self.qy),
            self.qw * self.qw + self.qx * self.qx - self.qy * self.qy - self.qz * self.qz
        )

        angle_roll = roll_rad * -57.296
        angle_pitch = pitch_rad * 57.296
        angle_yaw = yaw_rad * -57.296

        return angle_roll, angle_pitch, angle_yaw

    def estimatedGravityDirection(self):
        gx = 2 * (self.qx * self.qz - self.qw * self.qy)
        gy = 2 * (self.qw * self.qx + self.qy * self.qz)
        gz = self.qw * self.qw - self.qx * self.qx - self.qy * self.qy + self.qz * self.qz

        return gx, gy, gz

    def sensfusion6GetAccZ(self, ax, ay, az):
        """Return vertical acceleration"""
        # (A dot G) / |G|, (|G| = 1) -> (A dot G)
        return ax * self.gravX + ay * self.gravY + az * self.gravZ


if __name__ == '__main__':
    TWO_KP_DEF = 2.0 * 0.8
    TWO_KI_DEF = 2.0 * 0.001
    fusion = SensorFusion(TWO_KP_DEF, TWO_KI_DEF)
    fusion.sensfusion6UpdateQ(0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 0.01)
    print(fusion.sensfusion6GetEulerRPY())
    print(fusion.baseZacc)

from SensorFusion import *

TWO_KP_DEF = 2.0 * 0.8
TWO_KI_DEF = 2.0 * 0.001
toRad = 2*np.pi/360

previous_time = 0


def arrange_data(_line):
    """
    convert single string line in to csv data structure
    """
    _data = _line.split(',')
    _timestamp = int(_data[0])
    _elapsed_time = float(_data[1])
    _ax = float(_data[2])
    _ay = float(_data[3])
    _az = float(_data[4])
    _gx = float(_data[5])
    _gy = float(_data[6])
    _gz = float(_data[7])

    return _timestamp, _elapsed_time, _ax, _ay, _az, _gx, _gy, _gz


def euler_to_segment_angles(euler_angles):
    roll_rad, pitch_rad, yaw_rad = np.radians(euler_angles)

    rotation_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), np.sin(roll_rad)],
        [0, -np.sin(roll_rad), np.cos(roll_rad)]
    ])

    rotation_y = np.array([
        [np.cos(pitch_rad), 0, -np.sin(pitch_rad)],
        [0, 1, 0],
        [np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])

    rotation_z = np.array([
        [np.cos(yaw_rad), np.sin(yaw_rad), 0],
        [-np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    # Combine rotations in XYZ order
    rotation_matrix = rotation_x.dot(rotation_y).dot(rotation_z)

    segment_angles = \
        np.arctan2(rotation_matrix[0, 2], rotation_matrix[0, 0]),\
        np.arctan2(rotation_matrix[1, 2], rotation_matrix[1, 1]),\
        np.arctan2(rotation_matrix[0, 1], rotation_matrix[1, 1])

    return np.degrees(segment_angles)

def Resample_accelerometer(lowg, highg):
    """
    Function to align, fuse, and extract all IMU information.
    The high-g accelerometer is interpolated to the low-g accelerometer using
    the UNIX time-stamps in the innate files. The two accelerometer are then
    aligned using a cross correlation. Typically this results in a 1 frame
    phase shift.

    Parameters:
    lowg : pandas DataFrame
        low-g data table that is extracted as raw data from Capture.U. This
        table contains both the low-g accelerometer and the gyroscope_lowgoscope.
    highg : pandas DataFrame
        low-g data table that is extracted as raw data from Capture.U.

    Returns:
    lowgtime : numpy array (Nx1)
        time (UNIX) from the low-g file
    accelerometer : numpy array (Nx3)
        X,Y,Z fused (between low-g and high-g) accelerometer from the IMU
    gyroscope_lowg : numpy array (Nx3)
        X,Y,Z gyroscope_lowgoscope from the IMU
    """
    # lowg: low-g data table (contains accelerometer and gyroscope_lowgoscope)
    # highg: high-g data table (contains only accelerometer)

    # Need to align the data
    # 1: get the data collection frequencyuency from the low-g accelerometer
    accelerometer_lowg = lowg.iloc[:, 2:5].values
    gyroscope_lowg = lowg.iloc[:, 5:].values

    highgdata = highg.iloc[:, 2:].values

    highgtime = highg.iloc[:, 1].values
    lowgtime = lowg.iloc[:, 1].values

    index = (lowgtime < np.max(highgtime)) & (lowgtime > np.min(highgtime))
    lowgtime = lowgtime[index]
    accelerometer_lowg = accelerometer_lowg[index, :]
    gyroscope_lowg = gyroscope_lowg[index, :]

    # Create an empty array to fill with the resampled/downsampled high-g accelerometer
    resamplingHighg = np.zeros((len(lowgtime), 3))

    for jj in range(3):
        f = np.interp(lowgtime, highgtime, highgdata[:, jj])
        resamplingHighg[:, jj] = f

    # Cross-correlate the y-components
    corr_arr = np.correlate(accelerometer_lowg[:, 1], resamplingHighg[:, 1], mode='full')
    lags = np.arange(-len(accelerometer_lowg[:, 1])+1, len(resamplingHighg[:, 1]))
    lag = lags[np.argmax(corr_arr)]

    if lag > 0:
        lowgtime = lowgtime[lag+1:]
        gyroscope_lowg = gyroscope_lowg[lag+1:, :]
        accelerometer_lowg = accelerometer_lowg[lag+1:, :]
        resamplingHighg = resamplingHighg[:len(accelerometer_lowg), :]
    elif lag < 0:
        lowgtime = lowgtime[:len(lowgtime)+lag]
        gyroscope_lowg = gyroscope_lowg[:len(lowgtime), :]
        accelerometer_lowg = accelerometer_lowg[:len(lowgtime), :]
        resamplingHighg = resamplingHighg[-lag+1:, :]

    accelerometer = accelerometer_lowg

    # Find when the data is above/below 16G and replace with high-g accelerometer
    for jj in range(3):
        index = np.abs(accelerometer[:, jj]) > (9.81*16-0.1)
        accelerometer[index, jj] = resamplingHighg[index, jj]

    return lowgtime, accelerometer, gyroscope_lowg

def detect_jumps(accel_data, takeoff_threshold=20, flight_phase_threshold=-9.81, landing_threshold=40, min_flight_time=0.1, max_flight_time=0.5, sampling_rate=1125):
    min_samples_between_peaks = int(min_flight_time * sampling_rate)
    max_samples_between_peaks = int(max_flight_time * sampling_rate)
    samples_to_skip = 200
    
    i = 0
    takeoff_times = []
    landing_times = []
    takeoff_indices = []
    landing_indices = []
    peak_accelerations_takeoff = []
    peak_accelerations_landing = []
    flight_durations = []

    while i < len(accel_data) - 1:
        # Check for take-off condition
        if accel_data[i] > takeoff_threshold:
            start_index = i
            
            # Find the peak between this and the next value below the flight phase threshold
            while i < len(accel_data) - 1 and accel_data[i] > flight_phase_threshold:
                i += 1
            peak_takeoff = max(accel_data[start_index:i])
            potential_landing_start = i
            
            # Check within the time window for the next peak value above the landing threshold
            end_index = i + max_samples_between_peaks
            if end_index > len(accel_data):
                end_index = len(accel_data)
            potential_landing_data = accel_data[potential_landing_start:end_index]
            
            if any(val > landing_threshold for val in potential_landing_data):
                landing_peak_index = np.argmax(potential_landing_data) + potential_landing_start
                peak_landing = accel_data[landing_peak_index]
                
                time_diff = (landing_peak_index - start_index) / sampling_rate
                if min_flight_time <= time_diff <= max_flight_time:
                    # Store the results
                    takeoff_times.append(start_index / sampling_rate)
                    landing_times.append(landing_peak_index / sampling_rate)
                    takeoff_indices.append(start_index)
                    landing_indices.append(landing_peak_index)
                    peak_accelerations_takeoff.append(peak_takeoff)
                    peak_accelerations_landing.append(peak_landing)
                    flight_durations.append(time_diff)
                    i = landing_peak_index + samples_to_skip  # Skip ahead to avoid detections within 2 seconds
                    continue
            i += 1
        else:
            i += 1

    # Create a dataframe with the results
    jump_data = pd.DataFrame({
        'Takeoff Time (s)': takeoff_times,
        'Landing Time (s)': landing_times,
        'Takeoff Index': takeoff_indices,
        'Landing Index': landing_indices,
        'Peak Acceleration Takeoff (m/s^2)': peak_accelerations_takeoff,
        'Peak Acceleration Landing (m/s^2)': peak_accelerations_landing,
        'Flight Duration (s)': flight_durations
    })

    return jump_data


def trim_data(foot_accelerometer, foot_gyroscope, shank_accelerometer, shank_gyroscope, foot_time, shank_time):
    common_time_min = max(foot_time[0], shank_time[0])
    common_time_max = min(foot_time[-1], shank_time[-1])

    foot_time_trim = foot_time[(foot_time > common_time_min) & (foot_time < common_time_max)]
    shank_time_trim = shank_time[(shank_time > common_time_min)]
    shank_time_trim = shank_time[: len(foot_time_trim)]

    # Find the start and end index in the original data of foot
    foot_start_time_trimmed = foot_time_trim[0]
    foot_start_index_original = np.where(np.isclose(foot_time, foot_start_time_trimmed))[0][0]

    foot_end_time_trimmed = foot_time_trim[-1]
    foot_end_index_original = np.where(np.isclose(foot_time, foot_end_time_trimmed))[0][0]

    # Find the start and end index in the original data of shank
    shank_start_time_trimmed = shank_time_trim[0]
    shank_start_index_original = np.where(np.isclose(shank_time, shank_start_time_trimmed))[0][0]

    shank_end_time_trimmed = shank_time_trim[-1]
    shank_end_index_original = np.where(np.isclose(shank_time, shank_end_time_trimmed))[0][0]

    foot_accelerometer_trimmed = foot_accelerometer[foot_start_index_original:foot_end_index_original + 1]
    foot_gyroscope_trimmed = foot_gyroscope[foot_start_index_original:foot_end_index_original + 1]
    shank_accelerometer_trimmed = shank_accelerometer[shank_start_index_original:shank_end_index_original + 1]
    shank_gyroscope_trimmed = shank_gyroscope[shank_start_index_original:shank_end_index_original + 1]

    return foot_accelerometer_trimmed, foot_gyroscope_trimmed, shank_accelerometer_trimmed, shank_gyroscope_trimmed, foot_time_trim, shank_time_trim

def plot_angles(angle_list, title):
    """
    Plots the given angle list.
    angle_list: List of (n, 3) arrays for representing each jump angle.
    title: Title for the plots.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))  # 3 plots: for roll, pitch, and yaw
    
    # Assuming angles are stored as [roll, pitch, yaw]
    labels = ['Roll', 'Pitch', 'Yaw']
    colors = ['r', 'g', 'b']
    
    # Iterate over each axis (Roll, Pitch, Yaw)
    for i, ax in enumerate(axes):
        for j, jump in enumerate(angle_list):  # Iterate over each jump
            ax.plot(jump[:, i], color=colors[i], label=f"Jump {j+1}" if i == 0 else "")
        
        ax.set_ylabel(labels[i])
        ax.legend(loc='upper right')
        ax.grid(True)

    axes[-1].set_xlabel('Time (data points within jump)')
    plt.tight_layout()
    plt.suptitle(title, y=1.02)
    plt.show()

def csv_header():
    with open('EulerRPY.csv', 'a') as csv:
        csv.write('roll_deg')
        csv.write(',')
        csv.write('pitch_deg')
        csv.write(',')
        csv.write('yaw_deg')
        csv.write(',')
        csv.write('roll_segment')
        csv.write(',')
        csv.write('pitch_segment')
        csv.write(',')
        csv.write('yaw_segment')
        csv.write(',')
        csv.write('z_acceleration')
        csv.write(',\n')



shankhighg = pd.read_csv("/Users/ifeoluwaolawore/Library/CloudStorage/OneDrive-ILStateUniversity/Documents/BIOMECHANICS/SENSOR PROJECT/BOA Data/IMUData/S02-1-CMJ_TS-03391_2022-11-03-13-57-40_highg.csv")

shanklowg = pd.read_csv("/Users/ifeoluwaolawore/Library/CloudStorage/OneDrive-ILStateUniversity/Documents/BIOMECHANICS/SENSOR PROJECT/BOA Data/IMUData/S02-1-CMJ_TS-03391_2022-11-03-13-57-40_lowg.csv")

shankmag = pd.read_csv("/Users/ifeoluwaolawore/Library/CloudStorage/OneDrive-ILStateUniversity/Documents/BIOMECHANICS/SENSOR PROJECT/BOA Data/IMUData/S02-1-CMJ_TS-03391_2022-11-03-13-57-40_mag.csv")

foothighg = pd.read_csv("/Users/ifeoluwaolawore/Library/CloudStorage/OneDrive-ILStateUniversity/Documents/BIOMECHANICS/SENSOR PROJECT/BOA Data/IMUData/S02-1-CMJ_TS-03399_2022-11-03-14-04-23_highg.csv")

footlowg = pd.read_csv("/Users/ifeoluwaolawore/Library/CloudStorage/OneDrive-ILStateUniversity/Documents/BIOMECHANICS/SENSOR PROJECT/BOA Data/IMUData/S02-1-CMJ_TS-03399_2022-11-03-14-04-23_lowg.csv")

footmag = pd.read_csv("/Users/ifeoluwaolawore/Library/CloudStorage/OneDrive-ILStateUniversity/Documents/BIOMECHANICS/SENSOR PROJECT/BOA Data/IMUData/S02-1-CMJ_TS-03399_2022-11-03-14-04-23_mag.csv")


[flowgtime_untrimmed,faccelerometer_untrimmed,fgyroscope_lowg_untrimmed] = Resample_accelerometer(footlowg,foothighg)
[slowgtime_untrimmed,saccelerometer_untrimmed,sgyroscope_lowg_untrimmed] = Resample_accelerometer(shanklowg,shankhighg)

faccelerometer, fgyroscope_lowg, saccelerometer, sgyroscope_low, flowgtime, slowgtime = trim_data(faccelerometer_untrimmed,fgyroscope_lowg_untrimmed, saccelerometer_untrimmed,sgyroscope_lowg_untrimmed,flowgtime_untrimmed, slowgtime_untrimmed) 


if __name__ == '__main__':
    fusion = SensorFusion(TWO_KP_DEF, TWO_KI_DEF)
    
    sjump_angles = []  # This will be a list of arrays containing roll, pitch, yaw for each jump.
    ssegment_jump_angles = []  # This will be a list of arrays containing roll_segment, pitch_segment, yaw_segment for each jump.
    sjump_times = []  # This will store the time duration for each jump.
    
    accel_data = pd.Series(faccelerometer[:,2])
    accel_time = pd.Series(flowgtime)
    shank_accel_data = pd.Series(saccelerometer[:,0])
    shank_accel_time = pd.Series(slowgtime)
    
    detected_jumps = detect_jumps(accel_data, takeoff_threshold=20, flight_phase_threshold=-9.81, landing_threshold=40, min_flight_time=0.1, max_flight_time=0.5)
    ftakeoff_index = np.array(detected_jumps.iloc[:,2])
    flanding_index = np.array(detected_jumps.iloc[:,3])

    shank_detected_jumps = detect_jumps(shank_accel_data, takeoff_threshold=20, flight_phase_threshold=9.81, landing_threshold=40, min_flight_time=0.1, max_flight_time=1.0)
    stakeoff_index = np.array(shank_detected_jumps.iloc[:,2])
    slanding_index = np.array(shank_detected_jumps.iloc[:,3])
    
    saccelerometer = saccelerometer[:,[2,0,1]]
    sgyroscope_low = sgyroscope_low[:,[2,0,1]]
    
    for jump in range(len(ftakeoff_index)):
        takeoff_idx = stakeoff_index[jump]
        landing_idx = slanding_index[jump]

        current_jump_angles = []
        current_segment_angles = []

        for i in range(takeoff_idx, landing_idx + 1):  # We will iterate through each jump segment.
            dt = flowgtime[i] - flowgtime[i - 1] if i > 0 else 0
    
            fusion.sensfusion6UpdateQ(
                sgyroscope_low[i, 0], sgyroscope_low[i, 1], sgyroscope_low[i, 2],
                saccelerometer[i, 0], saccelerometer[i, 1], saccelerometer[i, 2], dt)
    
            roll, pitch, yaw = fusion.sensfusion6GetEulerRPY()
            roll_segment, pitch_segment, yaw_segment = euler_to_segment_angles([roll, pitch, yaw])
    
            # Storing all the angles for this segment.
            current_jump_angles.append([roll, pitch, yaw])
            current_segment_angles.append([roll_segment, pitch_segment, yaw_segment])
    
            # Optional: If you wish to print the angles.
            print(str("{:.3f}".format(roll)) + "\t" + str("{:.3f}".format(pitch)) + "\t" + str("{:.3f}".format(yaw)) +
                  "\t" + str("{:.3f}".format(roll_segment)) + "\t" + str("{:.3f}".format(pitch_segment)) +
                  "\t" + str("{:.3f}".format(yaw_segment)))
                
        sjump_angles.append(np.array(current_jump_angles))
        ssegment_jump_angles.append(np.array(current_segment_angles))
        
        # Calculate and store jump time duration
        sjump_times.append(flowgtime[landing_idx] - flowgtime[takeoff_idx])


plot_angles(sjump_angles, "Shank Angles for each Jump")
plot_angles(ssegment_jump_angles, " Shank Segment Angles for each Jump")

plot_angles(fjump_angles, "Foot Angles for each Jump")
plot_angles(fsegment_jump_angles, " Foot Segment Angles for each Jump")

    # print(fjump_angles)
    # print(fsegment_jump_angles)
    # print(jump_times)











# if __name__ == '__main__':
#     fusion = SensorFusion(TWO_KP_DEF, TWO_KI_DEF)
    
#     fjump_angles = []
#     print("Roll \t Pitch \t Yaw \t Roll_segment \t Pitch_segment \t Yaw_segment \t Z_Accel")
#     csv_header()

#     for i in range(len(flowgtime)):
#         dt = flowgtime[i] - flowgtime[i - 1] if i > 0 else 0

#         fusion.sensfusion6UpdateQ(
#             fgyroscope_lowg[i, 0], fgyroscope_lowg[i, 1], fgyroscope_lowg[i, 2],
#             faccelerometer[i, 0], faccelerometer[i, 1], faccelerometer[i, 2], dt)

#         roll, pitch, yaw = fusion.sensfusion6GetEulerRPY()
#         roll_segment, pitch_segment, yaw_segment = euler_to_segment_angles([roll, pitch, yaw])

#         z_acceleration = fusion.sensfusion6GetAccZ(faccelerometer[i, 0], faccelerometer[i, 1], faccelerometer[i, 2])

#         # comment out print line to execute faster
#         print(str("{:.3f}".format(roll)) + "\t" + str("{:.3f}".format(pitch)) + "\t" + str("{:.3f}".format(yaw)) +
#               "\t" + str("{:.3f}".format(roll_segment)) + "\t" + str("{:.3f}".format(pitch_segment)) +
#               "\t" + str("{:.3f}".format(yaw_segment)) + "\t" + str("{:.3f}".format(z_acceleration)))

#         with open('EulerRPY.csv', 'a') as txt:
#             txt.write(str("{:.3f}".format(roll)))
#             txt.write(',')
#             txt.write(str("{:.3f}".format(pitch)))
#             txt.write(',')
#             txt.write(str("{:.3f}".format(yaw)))
#             txt.write(',')
#             txt.write(str("{:.3f}".format(roll_segment)))
#             txt.write(',')
#             txt.write(str("{:.3f}".format(pitch_segment)))
#             txt.write(',')
#             txt.write(str("{:.3f}".format(yaw_segment)))
#             txt.write(',')
#             txt.write(str("{:.3f}".format(z_acceleration)))
#             txt.write(',\n')


