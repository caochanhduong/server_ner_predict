# -*- coding: utf-8 -*-
from proto import rest_api_pb2
from google.protobuf import json_format
from flask import Flask, jsonify, make_response, request as flask_request
from ner_activity import NERTAG
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
ner_model = NERTAG()
@app.route('/api/tagging/v1', methods=['POST'])
def sense():
    """V2 API, refere to proto/rest_api.proto for more detail."""
    try:
        proto_request = __extract_request(flask_request)
        response = __handle_request(proto_request)
    except Exception as ex:
        return make_response(__make_json_response_error(str(ex)), 500)

    return make_response(json_format.MessageToJson(response), 200)

def __handle_request(request):
    # TODO(minhta): Pass in model versions.
    response = rest_api_pb2.Response()
    
    mentions_to_rate = []
    
    for doc in request.items:
        mentions_to_rate.append(doc)

    results = ner_model.predict(mentions_to_rate)

    response.results.extend(results)
    
    return response

def __extract_request(flask_request):
    if not flask_request.is_json:
        raise ValueError('Expecting a json request.')

    parsed_pb = json_format.Parse(flask_request.get_data(),
                                  rest_api_pb2.Request())
    # Checks required fields.
    if not parsed_pb.items:
        raise ValueError('Expecting at least one document.')

    return parsed_pb

def __make_json_response_error(message):
    res = rest_api_pb2.Response()
    res.error.error_message = message
    return json_format.MessageToJson(res)

if __name__ == '__main__':
    app.run()