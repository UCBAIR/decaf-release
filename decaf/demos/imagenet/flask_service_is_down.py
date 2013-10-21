"""A simple interface to show when the flask server is down for debugging."""
import flask
from flask import Flask, url_for, request

# Obtain the flask app object
app = Flask(__name__)

@app.route('/')
def index():
    return "Decaf is down for debugging. Please check back in a few minutes."

if __name__ == '__main__':
    app.run(host='0.0.0.0')
