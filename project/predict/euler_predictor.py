import numba
import numpy as np

def calculate_path(ball, dt=0.01, max_steps=1000, g=np.array([0.0, 0.0, -9.81]), simple=True):
    """Calculate the path of the ball using Euler integration.
    
    Args:
        ball: An instance of the Ball class.
        dt (float): Time step for integration.
        max_steps (int): Maximum number of steps to simulate.
        g (np.ndarray): Gravity vector.
        simple (bool): If True, use simple Euler integration. If False, use a more complex method (not implemented yet).
    
    Returns:
        positions (np.ndarray): Array of shape (max_steps, 3) containing the positions
        velocities (np.ndarray): Array of shape (max_steps, 3) containing the velocities
    """
    p0 = ball.get_position()
    v0 = ball.get_velocity()

    if simple:
        # Calculate the time until the ball hits the ground (z=0) using the quadratic formula
        a, b, c = 0.5*g[2], v0[2], p0[2]
        t_impact = (-b - np.sqrt(b*b - 4*a*c)) / (2*a)
        n_valid = min(max_steps, int(np.ceil(t_impact/dt)))

        # Create arrays to hold the positions and velocities
        t = np.arange(n_valid) * dt
        positions = np.full((max_steps,3), np.nan)
        velocities = np.full((max_steps,3), np.nan)

        # Calculate positions and velocities using Euler integration
        positions[:n_valid] = p0 + v0*t[:,None] + 0.5*g*t[:,None]**2
        velocities[:n_valid] = v0 + g*t[:,None]
        
        return positions, velocities
    
    raise NotImplementedError("Complex integration method is not implemented yet.")

@numba.njit
def _loop(positions, velocities, p0, v0, g, dt, max_iter):
    cx, cy, cz = p0[0], p0[1], p0[2]
    cvx, cvy, cvz = v0[0], v0[1], v0[2]

    for i in range(max_iter):
        positions[i,0] = cx; positions[i,1] = cy; positions[i,2] = cz
        velocities[i,0] = cvx; velocities[i,1] = cvy; velocities[i,2] = cvz
       
        cx += cvx * dt
        cy += cvy * dt
        cz += cvz * dt
        
        cvx += g[0] * dt
        cvy += g[1] * dt
        cvz += g[2] * dt
       
        if cz <= 0:
            return i + 1  # Number of valid entries
    return max_iter

def calculate_path_numba(ball, dt=0.01, max_iter=1000, g=np.array([0.,0.,-9.81])):
    times = np.arange(max_iter) * dt
    positions = np.empty((max_iter,3))
    velocities = np.empty((max_iter,3))

    n = _loop(positions, velocities, ball.get_position(), ball.get_velocity(), g, dt, max_iter)
    
    return times[:n], positions[:n], velocities[:n]