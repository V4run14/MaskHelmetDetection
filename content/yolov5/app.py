from flask import Flask, render_template, Response
from detect1 import *

app = Flask(__name__)


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(detect(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host="10.11.20.95", port=8002)
    #app.run(debug=True)