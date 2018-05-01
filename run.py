# -*- encoding: utf-8 -*-

from cocoapp import app
from cocoapp.model2 import *

if __name__ == "__main__":
    #app.run()
    import cocoapp.model2
    app.run(host="0.0.0.0", debug=True, port=5000)
