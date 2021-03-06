import os
import tempfile
import numpy as np

def sobol_generate(n_dim, n_point, n_skip=0):
    if n_dim > 1111:
        raise Exception('This program supports sobol sequence of dimension up to 1111')
    while True:
        try:
            sequence_file = tempfile.NamedTemporaryFile('r')
            filename = sequence_file.name
            cmd = os.path.join(os.path.split(os.path.abspath(os.path.realpath(__file__)))[0], 'sobol_c/sobol')
            cmd += ' ' + str(int(n_dim)) + ' ' + str(int(n_point)) + ' ' + str(int(n_skip)) + ' ' + filename
            os.system(cmd)
            sequence = np.fromfile(sequence_file.file, dtype=np.float64).astype(np.float32)
            sequence = sequence.reshape([n_point, n_dim])
            sequence_file.close()
            return sequence
        except ValueError:
            print('%d data in file, but reshape is in (%d, %d)' % (sequence.size, n_point, n_dim))
            continue

if __name__ == '__main__':
    n_dim = 1111
    n_points = 10001
    n_dim = 128
    n_points = 64
    points = sobol_generate(n_dim, n_points, 0)
    print (np.min(points), np.max(points))
    print (points.shape)
    print (points.dtype)
    print (points[-1])
