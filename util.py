from sudoku import Sudoku
import gzip, cStringIO, cPickle, os
import exifread
import numpy as np

def solvepuzzle(puzzle):
    solver = Sudoku()
    solution = solver.solve(puzzle)
    return solver.stringify(solution)


def fast_unpickle_gzip(filepath):
    load_pkl = gzip.open(filepath, 'rb')
    memfile = cStringIO.StringIO()
    memfile.write(load_pkl.read())
    load_pkl.close()
    memfile.seek(0, os.SEEK_SET)
    obj = cPickle.load(memfile)
    memfile.close()
    return obj


def get_image_orientation(imbytes):
    # get orientation and rewind
    tags = exifread.process_file(imbytes, stop_tag='Orientation')
    imbytes.seek(0)
    return tags.get('Image Orientation')


def rotate_image(image, orientation=None):
    """
    apply rotation based on orientation tag
    """
    # apply orientation if any
    if orientation is not None:
        print orientation
        rot = {'Horizontal (normal)': 0, 'Rotated 180': 2,
               'Rotated 90 CCW': 3, 'Rotated 90 CW': 1}
        if orientation.printable in rot:
            return np.rot90(image, rot[orientation.printable])
        elif orientation.printable == 'Mirrored horizontal':
            return np.fliplr(image)
        elif orientation.printable == 'Mirrored vertical':
            return np.flipud(image)
        elif orientation.printable == 'Mirrored horizontal then rotated 90 CCW':
            return np.rot90(np.fliplr(image), 1)
        elif orientation.printable == 'Mirrored horizontal then rotated 90 CW':
            return np.rot90(np.fliplr(image), 3)
    return image