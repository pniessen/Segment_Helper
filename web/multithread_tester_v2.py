# multithread tester for Segment_Helper
# 4.10.2016 
# Don't forget to run mongod before starting this.

import flask
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for, send_file, copy_current_request_context
from pymongo import MongoClient
import os

# Initialize the Flask application
app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__)) + "/"

from logging import Formatter, FileHandler
handler = FileHandler(os.path.join(basedir, 'log3.txt'), encoding='utf8')
handler.setFormatter(
    Formatter("[%(asctime)s] %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S")
)
app.logger.addHandler(handler)

#MongoDB tools:
client   = MongoClient()
multi  = client.dsbc.multithread
multi.remove() # reset db
# cursor   = multi.find()
# review is a dict, e.g review = {}, then add key/value pairs
# multi.save(review)

def func_name():
    import traceback
    return traceback.extract_stack(None, 2)[0][2]

@app.route('/')
def index():
    with open("myproject.html", 'r') as viz_file:
        return viz_file.read()

# Route that will process the inbound data
@app.route('/inbound', methods=['POST'])
def inbound():
    z = func_name(); print "------in function:", z, "---------------"

    inbound_data = flask.request.json
    print inbound_data

    user_id = inbound_data['user_id']
    text = inbound_data['text']

    print user_id, text
    print type(user_id), type(text)

    double_me(user_id, text)

    return jsonify(user_id = user_id, text=text)

# http://stackoverflow.com/questions/11774265/flask-how-do-you-get-a-query-string-from-flask
@app.route('/outbound', methods=['GET'])
def outbound():
    z = func_name(); print "------in function:", z, "---------------"
    #user_id = flask.request.query_string
    user_id = flask.request.args.get('user_id')
    print user_id

    # text_list = results_dict[user_id]
    results = multi.find_one({'user_id':user_id})                                    
    print 'All documents: ', list(results)
    keys = list(results.keys())

    results = multi.find_one({'user_id':user_id})
    #y = list(results)[0]
    print 'All documents: ', list(results)
    results_list = [results[key] for key in keys if key not in [u'_id', u'user_id']]
    # text_list = list(results)[0].values()
    print type(results_list)

    outbound_data = {'text': results_list}

    return jsonify(outbound_data)

def double_me(user_id, text):
    z = func_name(); print "------in function:", z, "---------------"

    doubled_text = text + '_' + text

    multi.update_one({'user_id':user_id}, {'$set':{ text : doubled_text}},upsert=True)

    #if user_id not in results_dict.keys():
    #    results_dict[user_id] = {}

    #results_dict[user_id][text] = doubled_text
    results = multi.find_one({'user_id':user_id})
    print 'All documents: ', list(results)
    print 'results_db updated with ', len(list(results.keys())), ' keys!'
    # print results_dict[user_id]

    return


if __name__ == '__main__':
    #results_dict = {}
    #application.run(debug = True)
    app.run(host='0.0.0.0')
    #     port=int("80"),
    #     debug=True
    # 


