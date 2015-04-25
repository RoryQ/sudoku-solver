#!/usr/bin/env python

from base64 import b64encode
from datetime import datetime, timedelta
from json import dumps
from os import environ
from uuid import uuid4
import hmac, hashlib
import os.path

from flask import Flask, render_template, jsonify, make_response, request

ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png', 'gif', 'bmp'])

app = Flask(__name__)
for key in ('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_S3_BUCKET_URL'):
    app.config[key] = environ[key] 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/params', methods=['POST'])
def params():
    fname = request.json.get('filename')
    if not (fname and allowed_filetypes(fname)):
        return make_response(jsonify({'error' : '{0} file type not supported'.format(fname)}), 404)

    def make_policy():
        policy_object = {
                "expiration": (datetime.now() + timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                "conditions": [
                    {"bucket" : "dev.roryq.com"},
                    {"acl" : "public-read"},
                    ["starts-with", "$key", "seedoku/"],
                    {"success_action_status" : "201"}
                    ]
                }
        return b64encode(dumps(policy_object).replace('\n', '').replace('\r', ''))

    def sign_policy(policy):
        return b64encode(hmac.new(app.config['AWS_SECRET_ACCESS_KEY'], policy, hashlib.sha1).digest())

    policy = make_policy()
    return jsonify({
        "policy": policy,
        "signature": sign_policy(policy),
        "key": "seedoku/" + uuid4().hex + "." + get_extension(fname),
        "AWSAccessKeyId" : app.config['AWS_ACCESS_KEY_ID']
        })

def allowed_filetypes(filename):
    return '.' in filename and get_extension(filename) in ALLOWED_EXTENSIONS

def get_extension(filename):
    return os.path.splitext(filename)[1][1:].lower()

if __name__ == '__main__':
    app.run(debug=True)
