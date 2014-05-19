#!flask/bin/python
from flask import Flask, abort, jsonify, make_response, render_template, url_for, request
from forms import SodokuGrid
from sudoku import Sudoku
from jinja2 import evalcontextfilter, Markup, escape
from redis import Redis
from rq import Queue
from rq.job import Job
from functools import wraps
import os
import heroku
import re
import math
import urlparse

_paragraph_re = re.compile(r'(?:\r\n|\r|\n){2,}')

app = Flask(__name__)
app.config.from_object('config')

su = Sudoku()

redis_url = os.getenv('REDISTOGO_URL')
if not redis_url:
    raise RuntimeError('Set up Redis To Go first')

urlparse.uses_netloc.append('redis')
url = urlparse.urlparse(redis_url)
redis_conn = Redis(host=url.hostname, port=url.port, db=0, password=url.password)

q = Queue('default', connection=redis_conn)

heroku_key = os.getenv('HEROKU_API_KEY')

cloud = heroku.from_key(heroku_key)
app = cloud.apps['sudokusolver']

def hire(queue=None):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            ctx = f(*args, **kwargs)
            #hire
            if queue is not None and queue.count > 0:
                i = 0
                for a in app.processes['workers']:
                    i += 1
                if i == 0:
                    workers = math.ceil(queue.count/15.0)
                    app.processes['worker'].scale(workers)
            return ctx
        return decorated
    return decorator

def fire(queue=None):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            ctx = f(*args, **kwargs)
            #fire
            if queue is not None and queue.count == 0:
                app.processes['worker'].scale(0)
            return ctx
        return decorated
    return decorator

@app.route('/', methods=['POST', 'GET'])
@hire(q)
def index():
    form = SodokuGrid(csrf_enabled=False)
    if form.validate_on_submit():
        puzzle = form.data['Grid']
        if len(puzzle) != 81:
            abort(404)
        from util import solvepuzzle
        job = q.enqueue(solvepuzzle, puzzle)
        return render_template('processing.html',
                                job_id=job.id)
    return render_template('sudoku.html',
                           title="sudoku",
                           form=form)


@app.route('/processing/', methods=['POST'])
@fire(q)
def processing():
    job_id = request.form['job_id']
    job = Job(job_id, connection=redis_conn)
    if job.get_status() is None:
        return make_response(jsonify({'job_id': job_id, 'status': 'purged'}))
    if job.is_finished:
        resp = jsonify({'job_id': job_id,
                        'status': 'finished',
                        'url': url_for('rendergrid', puzzle=job.return_value)})
        return make_response(resp)
    return make_response(jsonify({'job_id': job.key, 'status': 'processing'}))

@app.route('/rendergrid/<string:puzzle>')
def rendergrid(puzzle=None):
    if puzzle is not None and len(puzzle) == 81 and su.solve(puzzle):
        display = su.display(su.solve(puzzle))
    else:
        abort(400)
    return render_template('puzzle.html',
                           puzzle=display)

@app.route('/api/v1.0/solve/<string:puzzle>', methods=['GET', 'POST'])
@hire(q)
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
