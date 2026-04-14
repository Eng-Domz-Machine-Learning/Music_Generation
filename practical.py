# Basic Perceptron Definition
def perceptron(weights, threshold, inputs):
    weighted_sum = sum(w * i for w, i in zip(weights, inputs))
    return 1 if weighted_sum >= threshold else 0


# Gate Definitions

def P_A_and_notX_and_Z(A, X, Z):
    return perceptron([1, -1, 1], 2, [A, X, Z])

def P_A_and_notY(A, Y):
    return perceptron([1, -1], 1, [A, Y])

def P_OR(x1, x2):
    return perceptron([1, 1], 1, [x1, x2])

def P_AND(x1, x2):
    return perceptron([1, 1], 2, [x1, x2])

def P_NOT(x):
    return perceptron([-1], 0, [x])


# Circuit Definition
def circuit(X, Y, Z, A):

    # Left side
    left1 = P_A_and_notX_and_Z(A, X, Z)
    left2 = P_A_and_notY(A, Y)
    Left = P_OR(left1, left2)

    # Right side
    right1 = P_AND(X, Z)
    X_and_Z = P_AND(X, Z)
    right2 = P_NOT(X_and_Z)
    Right = P_OR(right1, right2)

    OUT = P_AND(Left, Right)

    return OUT
