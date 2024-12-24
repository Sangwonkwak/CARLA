from filterpy.kalman import KalmanFilter
import numpy as np

# 처음부터 steering과 같은 스케일을 사용할거라 필요없음
# # 차 종류에 따라 steer값을 조향각으로 바꾸는 함수
# def steer_to_delta(steering, car_kind='something'):
#     return 1

def cal_theta(vx,vy):
    if vx > 0.0:
        return np.arctan(vy/vx)
    elif vx < 0.0:
        theta = np.arctan(vy/(-vx))
        theta = np.pi - theta if vy > 0.0 else -np.pi + theta
        return theta
    else:
        if vy == 0.0:
            return 0.0
        elif vy > 0.0:
            return np.pi / 2
        else:
            return np.pi / -2
        
class KFilter():
    def __init__(self,x,y,vx,vy,ax,ay,delta,dt):
        self.kf = None
        self.c_a = 0.00028 # 공기저항상수 (0.5*공기저항계수*차량의정면면적)/(차량의질량)
        self.w_accel = 0.01 # (throttle-brake)가 가속도에 미치는 영향
        self.dt = dt
        self.steering_threshold = 0.07 # 원래 코드는 0.1
        self.initialize_kf(x,y,vx,vy,ax,ay,delta)

    def set_dt(self, dt):
        self.dt = dt

    def initialize_kf(self,x,y,vx,vy,ax,ay,delta):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # [x,y,vx,vy,ax,ay,delta]
        self.kf.x = np.array([x,y,vx,vy,ax,ay,delta])
        # F
        self.kf.F = np.eye(7)
        # Process noise covariance
        state_dim = 7
        Q_std = [0.02,0.02,0.05,0.05,0.01,0.01,0.00001]
        Q = np.diag(np.square(Q_std))
        Q[0,2] = Q[2,0] = 0.0001 # x와 vx 상관관계
        Q[2,4] = Q[4,2] = 0.0001 # vx와 ax 상관관계
        Q[1,3] = Q[3,1] = 0.0001
        Q[3,5] = Q[5,3] = 0.0001
        self.kf.Q = Q
        # Q = np.random.normal(0,process_noise_std,(state_dim,state_dim))
        # Q = Q @ Q.T

        # measurement noise
        measurement_dim = 4
        measurement_noise_std = 0.01
        R_gnss = np.random.normal(0, measurement_noise_std, (measurement_dim,measurement_dim))
        R_gnss = R_gnss @ R_gnss.T
        mask = (np.arange(R_gnss.shape[0])[:, None] - np.arange(R_gnss.shape[1])) % 2 != 0 
        R_gnss[mask] = 0.0
        self.kf.R = R_gnss

        # Measurement function (GNSS) [x,y,vx,vy]
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]])
        # Initial state covariance
        P_std = [0.02,0.02,0.05,0.05,0.01,0.01,0.00001]
        P = np.diag(np.square(P_std))
        P[0,2] = P[2,0] = 0.0001 # x와 vx 상관관계
        P[2,4] = P[4,2] = 0.0001 # vx와 ax 상관관계
        P[1,3] = P[3,1] = 0.0001
        P[3,5] = P[5,3] = 0.0001
        self.kf.P = P
        # P = np.random.normal(0,P_std,(state_dim,state_dim))
        # P = P @ P.T
        
    
    def state_transition_matrix(self,vx,vy,dt,c_a):
        F = np.eye(7) # [x,y,vx,vy,ax,ay,delta]
        # x_pos
        F[0,2] = dt
        # y_pos
        F[1,3] = dt
        # vx
        F[2,2] = 1 - 2*c_a*vx*dt
        F[2,4] = dt
        # vy
        F[3,3] = 1 - 2*c_a*vy*dt
        F[3,5] = dt

        self.kf.F = F

    def predict_and_update(self,u,imu_data,z):
        throttle, brake, steering = u
        x, y, vx, vy, ax, ay, delta = self.kf.x
        imu_ax, imu_ay = imu_data
        self.kf.x = np.array([x,y,vx,vy,imu_ax,imu_ay,delta]) # imu의 가속도를 예측에 사용
        
        # update F
        self.state_transition_matrix(vx,vy,self.dt,self.c_a)
        # cal_theta(차량의 진행 방향 not head direction)
        theta = cal_theta(vx,vy)
        # Control input effects
        self.kf.B = np.zeros((7,2)) # control effect mapping
        delta_change = steering - delta
        if delta_change > self.steering_threshold:
            delta_change = self.steering_threshold
        elif delta_change < -self.steering_threshold:
            delta_change = -self.steering_threshold
        else:
            pass
        
        self.kf.B[2,0] = np.cos(theta) * self.dt
        self.kf.B[3,0] = np.sin(theta) * self.dt
        self.kf.B[6,1] = 1
        
        u_vec = np.array([self.w_accel*(throttle-brake),delta_change]) # control input
        # u_vec = u_vec.reshape(2,1) # 이거 하면 안된다
        # predict
        self.kf.predict(u=u_vec)
        # print(f'변화된 steer:{self.kf.x[-1]}')
        # update with measurements
        self.kf.update(z)
        # print(f'업데이트된 steer:{self.kf.x[-1]}')
        # print('########################################')
    
    def get_state(self):
        return self.kf.x
    
    def get_P(self):
        return self.kf.P



# # Run Kalman Filter
# kf = initialize_kf()
# u = np.array([throttle,brake,steering]) # 기존 방식에서 얻기
# imu_data = np.array([a,w]) # (t-1)시점 imu데이터
# z = np.array([x,y,psi,v]) # t시점 gnss데이터
# dt = carla_timestep
# c_a = 0.01 # normally 0.01 ~ 0.015
# w_accel = 1.0 # throttle과 brake가 가속도에 미치는 영향

# cur_state = np.zeros(7)
# P = np.zeros((7,7))
# cur_state, P = predict_and_update(kf,u,imu_data,z,dt,c_a,w_accel)

# print("Updated State: ", cur_state)










# def setup_kalman_filter(delta_t):
#     # state x is 6 dimensions, measurement z is 3 dimensions
#     kf = KalmanFilter(dim_x=6, dim_z=3)
#     # dynamics model F
#     kf.F = np.eye(6)
#     for x in range(3):
#         kf.F[x:x+3] = delta_t
#     # measurement matrix H
#     kf.H = np.array([
#         [1, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0],
#     ])

#     kf.x = np.ones(6)
    
#     # noise
#     kf.R *= 0.01 # sensor(measurement) noise
#     kf.Q = np.eyes(6,6) # process noise including external uncertainty
#     q = 0.1
#     weights = np.array([q,q,q,q/10,q/10,q/10])
#     kf.Q *= weights
#     kf.P *= 10 # covariance of initial state
#     return kf

# def kalman_update(kf,measurement):
#     kf.predict()
#     kf.update(measurement)
#     return kf.x[:3]