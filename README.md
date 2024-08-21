# Sensorfusion
This code focuses on processing and analyzing IMU (Inertial Measurement Unit) data from the foot and shank during jumps. Here's a breakdown of the key parts:

**1. Trim and Align Data:**
The trim_data function aligns the foot and shank IMU data by trimming both datasets to the common time range. It ensures both time-series data have the same duration.

**2. Data Resampling:**
The IMU data is first resampled using the Resample_accelerometer function to synchronize the high-g and low-g accelerometer readings for both the foot and shank.

**3. Jump Detection:**
The detect_jumps function identifies jumps in the acceleration data by applying thresholds for takeoff, flight, and landing phases.

**4. Angle Computation:**
The SensorFusion class (from an earlier part of the code) is used to compute roll, pitch, and yaw angles from the gyroscope and accelerometer data for both the foot and shank during each detected jump. These angles are also converted to "segment angles" (likely relative to a specific segment or axis).

**5. Visualization:**
The plot_angles function generates plots for the computed roll, pitch, and yaw angles for each jump. It creates separate plots for foot and shank angles, as well as for their segment equivalents.

**6. CSV Output:**
The csv_header function sets up a CSV file for storing the computed angles, allowing for further analysis or storage.

**7. Main Execution:**
In the if __name__ == '__main__': block, the code processes the IMU data, detects jumps, computes angles for each jump, and then plots these angles.
