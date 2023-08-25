"""This is the part of parameters setting in Pipeline Direction Algorithm
    |generalparams|
    - first_frame: the starting input of the algorithm
    - last_frame: the last frame input of the algorithm
    - current_frame: the current frame input of the algorithm (for debug mode)
    - frame_dir: the directory of dataset of a sequence pipeline images
    - range: FLS range specification
    - reinit_phi: the boolean of reinitial phi process (
                  estimate initial phi with skeleton of CA-CFAR)

    |preprocessingparams|
    - crop_frame_limit: [first row, last row, first column, last column]
                        for crop FLS image

    |multilookparams|
    - shift_time: the number of time that an image is shifted and
                then compute the correlation coefficient with the reference image

    |multilookparams|
    - phi_coef: a constant of initial phi
   """

class generalparams:
    first_frame = 1  # scene
    last_frame = 771  # stop_scene
    current_frame = 1  # figure
    frame_dir = ".\dataset\A"  # rootpath
    output_dir = ".\Results"
    crop_frame_limit = [25, 515, 10, 758]  # crop_lim
    max_row = 660
    max_col = 768
    reinit_phi = False  # re_phi


class preprocessingparams:
    range = 20  # range = 20 #r
    gaussian_win = 9
    gaussian_std = 3
    gaussian1d_win = 7


class multilookparams:
    shift_time = 10 #times

class levelsetparams:
    phi_coef = 2.0
    iteration_curved = 150
    iteration_straight = 200
    penalty_mew = 0.2 #mew
    length_lambda = 10.0 #lamda
    area_v = -1.0 #v
    GAC_alpha = 1.0
    epsilon = 1.0
    step_tau = 1.0 #t
    """initial phi process"""
    referenceCell_win = 51
    guardingCell_win = 41
    pfa = 0.3
    angle_limit_curved = [5, 70]
    angle_limit_straight = [75, 90]
    major_axis_length = 150 #minimum

class postprocessingparams:
    minimum_area = 20 #minimum which that segment is kept as a pipeline segment
    """curved shape"""
    length_threshold = 50 #if less than this -> Reinitial phi: True
    delta_x_threshold_curved = 50 #if less than or equal this -> Reinitial phi: True
    angle_threshold_curved = 85 #if more than this -> Reinitial phi: True
    """straight shape"""
    angle_threshold_straight = 85 #if less than or equal this -> Reinitial phi: True
