import numpy as np

def kalman_filter(raw, Q=1e-5, R=1e-2):
    n = len(raw)
    x_hat = np.zeros(n) # what we predict the raw data "should" be if there was no measurement noise
    P = np.zeros(n) # the uncertainties of our predicted noise-free data

    # initialization
    x_hat[0] = raw[0] # assume that the first measurement is what it should be
    P[0] = 1.0 # and that we make this assumption with 100% uncertainty

    for k in range(1,n): # for every subsequent measurement
        # predict what the next measurement should be, based on the previous k-1
        x_pred = x_hat[k-1]
        P_pred = P[k-1] + Q # our uncertainty increases since we propagate our uncertainty from k-1 to measurement k

        # update our prediction and uncertainty from k-1 based on what the measurement k actually was
        K = P_pred / (P_pred + R) # kalman gain determines our smoothing, based on R which is a parameter set to reflect the noisiness of the measurement
        x_hat[k] = x_pred + K * (raw[k] - x_pred)
        P[k] = (1 - K) * P_pred

    return x_hat