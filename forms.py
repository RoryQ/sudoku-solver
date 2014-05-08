from flask_wtf import Form
from wtforms import TextField


class SodokuGrid(Form):
    Grid = TextField()

