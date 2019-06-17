import os


def load_imu(path, sequence_idx):
    file_path = os.path.join(path, str(sequence_idx).zfill(4) + '.txt')
    assert os.path.exists(file_path), 'File does not exist: {}'.format(file_path)
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()

    imud = {i: [] for i in range(len(lines))}

    for idx, line in enumerate(lines):
        l = line.split(' ')
        _vf = float(l[8])  # Velocity forward
        _vl = float(l[9])  # Velocity leftward
        _vu = float(l[22]) # Rotation around upward
        imud[idx] = IMU(idx, _vf, _vl, _vu)

    return imud


class IMU:
    def __init__(self, frame, vf, vl, ru):
        self.frame = frame
        self.vf = vf    # Velocity forward
        self.vl = vl    # Velocity left
        self.ru = ru    # Rotation around up

    def __repr__(self):
        return '<IMU | Frame: {}, v_f = {} m/s, v_l = {} m/s, r_u = {} rad/s> \n'.format(self.frame, round(self.vf,2), round(self.vl,2), round(self.ru,2))
