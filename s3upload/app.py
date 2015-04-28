#!/usr/bin/env python

from base64 import b64encode
from datetime import datetime, timedelta
from json import dumps
from os import environ
from uuid import uuid4
from boto.s3.connection import S3Connection
import hmac, hashlib
import os.path

from flask import Flask, render_template, jsonify, make_response, request

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png', 'gif', 'bmp'])
SEEDOKU_BUCKET = 'seedoku.roryq.com'

app = Flask(__name__)
for key in ('AWS_SEEDOKU_WRITE_KEY', 'AWS_SEEDOKU_WRITE_SECRET', 'AWS_S3_BUCKET_URL',
        'AWS_SEEDOKU_READ_KEY', 'AWS_SEEDOKU_READ_SECRET'):
    app.config[key] = environ[key] 


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/public_link')
def public_link():
    key = request.args.get('key')
    s3conn = S3Connection(app.config['AWS_SEEDOKU_WRITE_KEY'], app.config['AWS_SEEDOKU_WRITE_SECRET'])
    url = s3conn.generate_url(300, 'GET', SEEDOKU_BUCKET, key)
    print key
    return url

@app.route('/params', methods=['POST'])
def params():
    fname = request.json.get('filename')
    if not (fname and allowed_filetypes(fname)):
        return make_response(jsonify({'error' : '{0} file type not supported'.format(fname)}), 404)

    def make_policy():
        policy_object = {
                "expiration": (datetime.now() + timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                "conditions": [
                    {"bucket" : SEEDOKU_BUCKET},
                    {"acl" : "authenticated-read"},
                    ["starts-with", "$key", "seedoku/"],
                    {"success_action_status" : "201"}
                    ]
                }
        return b64encode(dumps(policy_object).replace('\n', '').replace('\r', ''))

    def sign_policy(policy):
        return b64encode(hmac.new(app.config['AWS_SEEDOKU_WRITE_SECRET'], policy, hashlib.sha1).digest())

    policy = make_policy()
    return jsonify({
        "policy": policy,
        "signature": sign_policy(policy),
        "key": "seedoku/" + uuid4().hex + "." + get_extension(fname),
        "AWSAccessKeyId" : app.config['AWS_SEEDOKU_WRITE_KEY']
        })

def allowed_filetypes(filename):
    return '.' in filename and get_extension(filename) in ALLOWED_EXTENSIONS

def get_extension(filename):
    return os.path.splitext(filename)[1][1:].lower()

if __name__ == '__main__':
    app.run(debug=True)
