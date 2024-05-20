import numpy as np

# 특정 주파수에서의 임피던스를 계산하는 함수
def calculate_impedance(R, L, C, frequency):
    omega = 2 * np.pi * frequency
    Z_R = R
    Z_L = 1j * omega * L
    Z_C = 1 / (1j * omega * C)
    Z_total = Z_R + Z_L + Z_C
    return np.abs(Z_total)

# 데이터 생성 함수
def generate_data(num_samples, frequency):
    R_values = np.random.uniform(1, 1000, num_samples)
    L_values = np.random.uniform(1e-6, 1e-3, num_samples)
    C_values = np.random.uniform(1e-12, 1e-6, num_samples)
    impedances = []

    for R, L, C in zip(R_values, L_values, C_values):
        impedance = calculate_impedance(R, L, C, frequency)
        impedances.append(impedance)
    
    return np.array(R_values), np.array(L_values), np.array(C_values), np.array(impedances)

# 예시 주파수 설정
frequency = 1e3  # 1e3은 1 kHz를 의미
R_values, L_values, C_values, impedances = generate_data(1000, frequency)



# 손실 함수
def loss_function(R, L, C, target_impedance, frequency):
    impedance = calculate_impedance(R, L, C, frequency)
    loss = (impedance - target_impedance) ** 2
    return loss


# 경사 계산 함수
def compute_gradients(R, L, C, target_impedance, frequency):
    delta = 1e-9
    loss = loss_function(R, L, C, target_impedance, frequency)

    dR = (loss_function(R + delta, L, C, target_impedance, frequency) - loss) / delta
    dL = (loss_function(R, L + delta, C, target_impedance, frequency) - loss) / delta
    dC = (loss_function(R, L, C + delta, target_impedance, frequency) - loss) / delta

    return dR, dL, dC



# 경사하강법 최적화 함수
def gradient_descent(R, L, C, target_impedance, frequency, learning_rate, iterations):
    for _ in range(iterations):
        dR, dL, dC = compute_gradients(R, L, C, target_impedance, frequency)
        R -= learning_rate * dR
        L -= learning_rate * dL
        C -= learning_rate * dC
        print(f'Loss: {loss_function(R, L, C, target_impedance, frequency)}, R: {R}, L: {L}, C: {C}')
    return R, L, C

# 초기값 설정
R_initial, L_initial, C_initial = 100, 1e-6, 1e-9
learning_rate = 1e-3
iterations = 1000
target_impedance = 50  # 목표 임피던스 <- 우리는 이걸 맞추기 위해서 계속 경사하강법을 진행할 것이다.

# 최적화 실행
R_optimal, L_optimal, C_optimal = gradient_descent(R_initial, L_initial, C_initial, target_impedance, frequency, learning_rate, iterations)
print(f'Optimal R: {R_optimal}, Optimal L: {L_optimal}, Optimal C: {C_optimal}')
