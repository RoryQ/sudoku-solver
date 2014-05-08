#!flask/bin/python
from flask import Flask, abort, jsonify, make_response, render_template, redirect, url_for
from forms import SodokuGrid
from sudoku import Sudoku
from jinja2 import evalcontextfilter, Markup, escape

import re

_paragraph_re = re.compile(r'(?:\r\n|\r|\n){2,}')

app = Flask(__name__)
app.config.from_object('config')

su = Sudoku()

@app.route('/', methods=['POST', 'GET'])
def index():
    form = SodokuGrid(csrf_enabled=False)
    if form.validate_on_submit():
        puzzle = form.data['Grid']
        if len(puzzle) != 81:
            abort(404)
        solution = su.solve(puzzle)
        return redirect(url_for('rendergrid',
                                puzzle=su.stringify(solution)))
    return render_template('sudoku.html',
                           title="sudoku",
                           form=form)

@app.route('/rendergrid/<string:puzzle>')
def rendergrid(puzzle=None):
    if puzzle is not None and len(puzzle) == 81 and su.solve(puzzle):
        display = su.display(su.solve(puzzle))
    else:
        abort(400)
    return render_template('puzzle.html',
                           puzzle=display)

@app.route('/api/v1.0/solve/<string:puzzle>', methods=['GET', 'POST'])
def solve_puzzle(puzzle):
    if len(puzzle) != 81:
        abort(1404)
    solution = su.solve(puzzle)
    display = ""
    if solution is not False:
        display = su.display(solution)
    return jsonify({'solution': solution, 'display': display, 'puzzle': puzzle})

@app.errorhandler(1404)
def not_found(error):
    return make_response(jsonify({'error': 'not found'}), 404)

@app.template_filter()
@evalcontextfilter
def nl2br(eval_ctx, value):
    result = u'\n\n'.join(u'<p>%s</p>' % p.replace('\n', Markup('<br/>\n')) \
        for p in _paragraph_re.split(escape(value)))
    if eval_ctx.autoescape:
        result = Markup(result)
    return result

if __name__ == '__main__':
    app.run(debug=True)
