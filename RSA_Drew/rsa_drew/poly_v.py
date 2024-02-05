import numpy as np

def PolyV(degree, num_points):

    """
    Generates polynomial values based on the given degree and number of points

    Parameters:
    degree (int): Degree of the polynomial
    num_points (int): Number of points for which polynomial values are generated

    Returns:
    array: Array of polynomial values
    """
    
    polyvals = np.zeros(num_points)

    if degree < 2 or degree > 5:
        raise ValueError("Degree must be between 2 and 5")

    if degree + 1 > num_points:
        raise ValueError("Number of points must be greater than degree + 1")

    # Adjusting num_points to the closest even number less than num_points
    if num_points % 2 != 0:
        num_points -= 1

    n3 = num_points // 2
    a0 = num_points
    a2 = a4 = a6 = a8 = 0

    for i in range(1, n3 + 1):
        aI = i
        a2 += 2 * aI ** 2
        a4 += 2 * aI ** 4
        a6 += 2 * aI ** 6
        a8 += 2 * aI ** 8

    aJ = -n3 - 1
    if degree > 3:
        den = 1.0 / (a0 * a4 * a8 + 2 * a2 * a4 * a6 - a4 ** 3 - a0 * a6 ** 2 - a2 ** 2 * a8)
        c1 = a4 * a8 - a6 ** 2
        c2 = a4 * a6 - a2 * a8
        c3 = a2 * a6 - a4 ** 2

        for i in range(num_points):
            aJ += 1
            polyvals[i] = den * (c1 + c2 * aJ ** 2 + c3 * aJ ** 4)
    else:
        den = 1.0 / (a0 * a4 - a2 ** 2)
        for i in range(num_points):
            aJ += 1
            aJ2 = aJ ** 2
            polyvals[i] = den * (a4 - aJ2 * a2)

    # symmetrize the polynomial values
    for i in range(n3 + 1, num_points):
        polyvals[i] = polyvals[num_points - i]

    # normalize the polynomial values
    polyvals /= np.sum(polyvals)

    return polyvals

# pre-calculated poly values using original MATLAB script
PRE_CALCULATED_DEG_3_SIZE_51 = [
    -0.0266,
    -0.0211,
    -0.0158,
    -0.0107,
    -0.0058,
    -0.0012,
    0.0033,
    0.0075,
    0.0114,
    0.0152,
    0.0187,
    0.0219,
    0.0250,
    0.0278,
    0.0304,
    0.0328,
    0.0350,
    0.0369,
    0.0386,
    0.0401,
    0.0413,
    0.0423,
    0.0431,
    0.0437,
    0.0440,
    0.0441,
    0.0440,
    0.0437,
    0.0431,
    0.0423,
    0.0413,
    0.0401,
    0.0386,
    0.0369,
    0.0350,
    0.0328,
    0.0304,
    0.0278,
    0.0250,
    0.0219,
    0.0187,
    0.0152,
    0.0114,
    0.0075,
    0.0033,
    -0.0012,
    -0.0058,
    -0.0107,
    -0.0158,
    -0.0211,
    -0.0266
]