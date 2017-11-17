from flask import jsonify
from werkzeug.exceptions import HTTPException

class JsonResponse:

    def make_json_error(self, ex):
        response = jsonify(message=str(ex))
        response.status_code = (ex.code
                                if isinstance(ex, HTTPException)
                                else 500)
        return response

    # TODO: create a method to return a complete response object