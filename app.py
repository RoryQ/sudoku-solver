#!flask/bin/python
from flask import Flask, abort, jsonify, make_response, render_template, url_for, request
from sudoku import Sudoku
from jinja2 import evalcontextfilter, Markup, escape
import os
import re
from json import dumps
from os import environ
from uuid import uuid4
import hmac
import hashlib
import os.path
from base64 import b64encode
from datetime import datetime, timedelta
from run_celery import make_celery
from SeedokuCeleryTask import SeedokuTask
from celery.exceptions import TimeoutError


_paragraph_re = re.compile(r'(?:\r\n|\r|\n){2,}')
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png', 'gif', 'bmp'])

flask_app = Flask(__name__)
flask_app.config.from_object('config')
flask_app.debug = os.getenv('DEBUG') == "True"
for key in ('AWS_SEEDOKU_WRITE_KEY', 'AWS_SEEDOKU_WRITE_SECRET',
            'AWS_S3_BUCKET_URL', 'AWS_SEEDOKU_READ_KEY',
            'AWS_SEEDOKU_READ_SECRET', 'AWS_SEEDOKU_S3_BUCKET',
            'CELERY_BROKER_URL', 'CELERY_RESULT_BACKEND',
            'TESSERACT_LANG', 'TESSERACT_LIBPATH', 'TESSERACT_TESSDATA'):
    flask_app.config[key] = environ[key]

celery_app = make_celery(flask_app)

su = Sudoku()

seedoku = SeedokuTask(flask_app.config)


@celery_app.task(name="tasks.async_image_to_puzzle")
def async_image_to_puzzle(key):
    return seedoku.aws_upload_key_to_puzzle(key)


@flask_app.route('/', methods=['GET', 'POST'])
def upload_to_s3():
    if request.method == 'POST':
        key = request.json.get('key')
        res = async_image_to_puzzle.apply_async((key,))
        return jsonify(task_id=res.task_id)
    return render_template('s3upload.html')


@flask_app.route('/processing', methods=['POST'])
def processing():
    task_id = request.json.get('task_id')
    state = async_image_to_puzzle.AsyncResult(task_id).state
    print state
    try:
        result = async_image_to_puzzle.AsyncResult(task_id).get(timeout=1)
        render = render_template('puzzle_snippet.html',
                                 puzzle=su.display(result[1]))
    except TimeoutError:
        result = None
        render = None
    return make_response(jsonify({"status": state, "render": render}))


@flask_app.route('/s3_config_params', methods=['POST'])
def s3_config_params():
    fname = request.json.get('filename')
    if not (fname and allowed_filetype(fname)):
        return make_response(jsonify(
            {'error': '{0} file type not supported'.format(fname)}), 400)

    def make_policy():
        policy_object = {
                "expiration": (datetime.now() + timedelta(hours=1))
                .strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                "conditions": [
                    {"bucket": flask_app.config['AWS_SEEDOKU_S3_BUCKET']},
                    {"acl": "authenticated-read"},
                    ["starts-with", "$key", "seedoku/"],
                    {"success_action_status": "201"}
                    ]
                }
        policy = dumps(policy_object).replace('\n', '').replace('\r', '')
        return b64encode(policy)

    def sign_policy(policy):
        return b64encode(hmac.new(flask_app.config['AWS_SEEDOKU_WRITE_SECRET'],
                                  policy,
                                  hashlib.sha1).digest())

    policy = make_policy()
    return jsonify({
        "policy": policy,
        "signature": sign_policy(policy),
        "key": "seedoku/" + uuid4().hex + "." + get_extension(fname),
        "AWSAccessKeyId": flask_app.config['AWS_SEEDOKU_WRITE_KEY']
        })


@flask_app.errorhandler(1404)
def not_found(error):
    return make_response(jsonify({'error': 'not found'}), 404)


@flask_app.template_filter()
@evalcontextfilter
def nl2br(eval_ctx, value):
    result = u'\n\n'.join(u'<p>%s</p>' % p.replace('\n', Markup('<br/>\n'))
                          for p in _paragraph_re.split(escape(value)))
    if eval_ctx.autoescape:
        result = Markup(result)
    return result


def allowed_filetype(filename):
    return '.' in filename and get_extension(filename) in ALLOWED_EXTENSIONS


def get_extension(filename):
    return os.path.splitext(filename)[1][1:].lower()

if __name__ == '__main__':
    flask_app.run(debug=True)
