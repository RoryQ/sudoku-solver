from sudoku import Sudoku
import gzip, cStringIO, cPickle, os

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
