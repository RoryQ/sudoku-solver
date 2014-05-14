from sudoku import Sudoku

def solvepuzzle(puzzle):
    solver = Sudoku()
    solution = solver.solve(puzzle)
    return solver.stringify(solution)
