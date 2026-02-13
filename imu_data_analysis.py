import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


path = "/Users/chrisgnam/Desktop/imu_data.csv"


def main():
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig_accel, ax_accel = plt.subplots(figsize=(10, 4))
    ax_accel.plot(df["timestamp"], df["ax"], label="ax")
    ax_accel.plot(
        df["timestamp"],
        df["ax"].rolling(window=500, min_periods=1).mean(),
        label="ax rolling mean (500)",
    )
    # ax_accel.plot(df["timestamp"], df["ay"], label="ay")
    #ax_accel.plot(df["timestamp"], df["az"], label="az")
    ax_accel.set_title("Acceleration")
    ax_accel.set_xlabel("Time")
    ax_accel.set_ylabel("m/s^2")
    ax_accel.legend()
    fig_accel.autofmt_xdate()

    fig_gyro, ax_gyro = plt.subplots(figsize=(10, 4))
    # ax_gyro.plot(df["timestamp"], df["wx"], label="wx")
    # ax_gyro.plot(df["timestamp"], df["wy"], label="wy")
    ax_gyro.plot(df["timestamp"], df["wz"], label="wz")

    print("Standard deviation of wz:", np.std(df["wz"]))

    ax_gyro.plot(
        df["timestamp"],
        df["wz"].rolling(window=500, min_periods=1).mean(),
        label="wz rolling mean (500)",
    )
    ax_gyro.set_title("Angular Rate")
    ax_gyro.set_xlabel("Time")
    ax_gyro.set_ylabel("rad/s")
    ax_gyro.legend()
    fig_gyro.autofmt_xdate()

    plt.show()

    


if __name__ == "__main__":
    main()