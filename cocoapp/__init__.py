from flask import Flask

app = Flask(__name__)
app.config.from_object('cocoapp.default_settings')

import cocoapp.views
