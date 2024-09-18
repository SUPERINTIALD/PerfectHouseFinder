from typing import Optional
from flask import Flask, abort, redirect, request, render_template, session
# from database.database import Database
import random
app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/home')
def home():
	return render_template('home.html')

if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)