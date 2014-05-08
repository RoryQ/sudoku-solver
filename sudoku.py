## Solve Every Sudoku Puzzle

## See http://norvig.com/sudoku.html

## Throughout this program we have:
##   r is a row,    e.g. 'A'
##   c is a column, e.g. '3'
##   s is a square, e.g. 'A3'
##   d is a digit,  e.g. '9'
##   u is a unit,   e.g. ['A1','B1','C1','D1','E1','F1','G1','H1','I1']
##   grid is a grid,e.g. 81 non-blank chars, e.g. starting with '.18...7...
##   values is a dict of possible values, e.g. {'A1':'12349', 'A2':'8', ...}


class Sudoku():

    digits = '123456789'
    rows = 'ABCDEFGHI'
    cols = digits

    def __init__(self):
        self.squares = self._cross(self.rows, self.cols)
        self.unitlist = ([self._cross(self.rows, c) for c in self.cols] +
                    [self._cross(r, self.cols) for r in self.rows] +
                    [self._cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')])
        self.units = dict((s, [u for u in self.unitlist if s in u])
                          for s in self.squares)
        self.peers = dict((s, set(sum(self.units[s], []))-set([s]))
                          for s in self.squares)

    def _cross(self, A, B):
        """Cross product of elements in A and elements in B."""
        return [a+b for a in A for b in B]

    def _parse_grid(self, grid):
        """Convert grid to a dict of possible values, {square: digits}, or
        return False if a contradiction is detected."""
        ## To start, every square can be any digit; then assign values from the grid.
        values = dict((s, self.digits) for s in self.squares)
        for s, d in self._grid_values(grid).items():
            if d in self.digits and not self._assign(values, s, d):
                return False  # (Fail if we can't assign d to square s.)
        return values

    def _grid_values(self, grid):
        """Convert grid into a dict of {square: char} with '0' or '.' for empties."""
        chars = [c for c in grid if c in self.digits or c in '0.']
        assert len(chars) == 81
        return dict(zip(self.squares, chars))

    def _assign(self, values, s, d):
        """Eliminate all the other values (except d) from values[s] and propagate.
        Return values, except return False if a contradiction is detected."""
        other_values = values[s].replace(d, '')
        if all(self._eliminate(values, s, d2) for d2 in other_values):
            return values
        else:
            return False

    def _eliminate(self, values, s, d):
        """Eliminate d from values[s]; propagate when values or places <= 2.
        Return values, except return False if a contradiction is detected."""
        if d not in values[s]:
            return values  # Already eliminated
        values[s] = values[s].replace(d,'')
        ## (1) If a square s is reduced to one value d2, then eliminate d2 from the peers.
        if len(values[s]) == 0:
            return False  # Contradiction: removed last value
        elif len(values[s]) == 1:
            d2 = values[s]
            if not all(self._eliminate(values, s2, d2) for s2 in self.peers[s]):
                return False
        ## (2) If a unit u is reduced to only one place for a value d, then put it there.
        for u in self.units[s]:
            dplaces = [s for s in u if d in values[s]]
            if len(dplaces) == 0:
                return False  # Contradiction: no place for this value
            elif len(dplaces) == 1:
                # d can only be in one place in unit; assign it there
                if not self._assign(values, dplaces[0], d):
                    return False
        return values

################ Display as 2-D grid ################

    def display(self, values):
        """Display these values as a 2-D grid."""
        width = 1+max(len(values[s]) for s in self.squares)
        line = '+'.join(['-'*(width*3)]*3)
        l = ""
        for r in self.rows:
            l += ''.join(values[r+c].center(width)+('|' if c in '36' else '')
                        for c in self.cols)
            if r in 'CF':
                l += "\n" + line
            l += "\n"
        return l

    def stringify(self, values):
        """Convert values into 81 char string"""
        l = ""
        for r in self.rows:
            l += ''.join(values[r+c] for c in self.cols)
        return l

################ Search ################

    def solve(self, grid):
        return self._search(self._parse_grid(grid))

    def _search(self, values):
        """Using depth-first search and propagation, try all possible values."""
        if values is False:
            return False # Failed earlier
        if all(len(values[s]) == 1 for s in self.squares):
            return values # Solved!
        ## Chose the unfilled square s with the fewest possibilities
        n, s = min((len(values[s]), s) for s in self.squares if len(values[s]) > 1)
        return self.some(self._search(self._assign(values.copy(), s, d))
                         for d in values[s])

################ Utilities ################

    def some(self, seq):
        """Return some element of seq that is true."""
        for e in seq:
            if e:
                return e
        return False

## References used:
## http://www.scanraid.com/BasicStrategies.htm
## http://www.sudokudragon.com/sudokustrategy.htm
## http://www.krazydad.com/blog/2005/09/29/an-index-of-sudoku-strategies/
## http://www2.warwick.ac.uk/fac/sci/moac/currentstudents/peter_cock/python/sudoku/