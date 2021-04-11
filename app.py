from flask import Flask
from flask import request

from model import *

app = Flask(__name__)

version=10

@app.before_first_request
def before_first_request():
    init()

@app.route('/recommend', methods=['GET'])
def recommend():
	token = request.args.get('token')
	print(f"LOG: Input token is {token}")
	return get_recommendations(token)
	
@app.route('/health', methods=['GET'])
def healthcheck():
	print(f"LOG: Version is {version}")
	return f"Version is {version}"
	
if __name__ == "__main__":
    app.run()