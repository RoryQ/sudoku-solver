#!flask/bin/python
from flask import Flask, abort, jsonify, make_response, render_template, url_for, request
from forms import SodokuGrid
from sudoku import Sudoku
from jinja2 import evalcontextfilter, Markup, escape
import redis
from rq import Queue
from rq.job import Job
from functools import wraps
import os
import heroku
import re
import math
import util
from seedoku import Seedoku
import cv2

_paragraph_re = re.compile(r'(?:\r\n|\r|\n){2,}')

app = Flask(__name__)
app.config.from_object('config')
app.debug = os.getenv('DEBUG') == "True"

su = Sudoku()

ocr = util.fast_unpickle_gzip('SVM.p.gz')
ocr._update_rq = False
print 'ocr loaded'

redis_url = os.getenv('REDISTOGO_URL', 'redis://localhost:6379')
redis_conn = redis.from_url(redis_url)

q = Queue('default', connection=redis_conn)

heroku_key = os.getenv('HEROKU_API_KEY')

cloud = heroku.from_key(heroku_key)
heroku_app = cloud.apps['sudokusolver']

def hire(queue=None):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            ctx = f(*args, **kwargs)
            #hire
            if queue is not None and queue.count > 0:
                i = 0
                try:
                    for a in heroku_app.processes['worker']:
                        i += 1
                except KeyError:
                    i = 0
                if i == 0:
                    workers = int(math.ceil(queue.count/15.0))
                    #heroku_app.processes['worker'].scale(workers)
                    cloud._http_resource(
                            method='POST',
                            resource=('apps', 'sudokusolver', 'ps', 'scale'),
                            data={'type': 'worker', 'qty': workers})
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
                #heroku_app.processes['worker'].scale(0)
                cloud._http_resource(
                        method='POST',
                        resource=('apps', 'sudokusolver', 'ps', 'scale'),
                        data={'type': 'worker', 'qty': 0})
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
        return render_template('processing.html', job_id=job.id)
    return render_template('sudoku.html', title="sudoku", form=form)


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
    return render_template('puzzle.html', puzzle=display)

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

@app.route('/seedoku/', methods=['GET', 'POST'])
def seedoku_test():
    seedoku = Seedoku(ocr)
    img = cv2.imread('puzzle.jpg')
    sol = seedoku.image_to_puzzle(img)
    print sol
    return render_template('puzzle.html', puzzle=su.display(sol))

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
