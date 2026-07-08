from scipy.spatial.transform import Rotation

def calc_orientation_error(current_rotation, desired_rotation, return_as_rotvec=False):
    """Compute the orientation error between two rotation matrices.

    Args:
        current_rotation (np.ndarray): Current rotation matrix (3x3).
        desired_rotation (np.ndarray): Desired rotation matrix (3x3).
        return_as_rotvec (bool): If True, return as a rotation vector. If False, return as a rotation matrix.

    Returns:
        np.ndarray: Orientation error as a rotation matrix (3x3) or rotation vector (3,).
    """

    # Create rotation objects
    rot_curr = Rotation.from_matrix(current_rotation)
    rot_des = Rotation.from_matrix(desired_rotation)
    
    # Compute the relative rotation
    relative_rot = rot_des * rot_curr.inv()
    
    if return_as_rotvec:
        return relative_rot.as_rotvec()
    else:
        return relative_rot.as_matrix()