from flask_restplus import Namespace, Resource, fields
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import BadRequest
from config import MODEL_META_DATA
from core.backend import ModelWrapper
import os
import random

api = Namespace('model', description='Model information and inference operations')

model_meta = api.model('ModelMetadata', {
    'id': fields.String(required=True, description='Model identifier'),
    'name': fields.String(required=True, description='Model name'),
    'description': fields.String(required=True, description='Model description'),
    'license': fields.String(required=False, description='Model license')
})


@api.route('/metadata')
class Model(Resource):
    @api.doc('get_metadata')
    @api.marshal_with(model_meta)
    def get(self):
        """Return the metadata associated with the model"""
        return MODEL_META_DATA


predict_response = api.model('ModelPredictResponse', {
    'status': fields.String(required=True, description='Response status message'),
    'embedding': fields.List(fields.List(fields.Float, required=True, description="Generated embedding"))
})

# set up parser for audio input data
audio_parser = api.parser()
audio_parser.add_argument('audio', type=FileStorage, location='files', required=True,
                          help="signed 16-bit PCM WAV audio file")


@api.route('/predict')
class Predict(Resource):

    @api.doc('predict')
    @api.expect(audio_parser)
    @api.marshal_with(predict_response)
    def post(self):
        try:
            mw = ModelWrapper()
            """Generate audio embedding from input data"""
            result = {'status': 'error'}

            args = audio_parser.parse_args()
            audio_data = args['audio'].read()

            new_file_name = '{}_{}.wav'.format(args['audio'], random.randint(1,10000000000))
            # clean up from earlier runs
            if os.path.exists(new_file_name):
                os.remove(new_file_name)

            if '.wav' in str(args['audio']):
                file = open(new_file_name, "wb")
                file.write(new_file_name)
                file.close()
            else:
                e = BadRequest()
                e.data = {'status': 'error', 'message': 'Invalid file type/extension'}
                raise e

            # Getting the predictions
            preds = mw.predict(new_file_name)

            # Aligning the predictions to the required API format
            result['embedding'] = preds.tolist()
            result['status'] = 'ok'

            os.remove(new_file_name)

            return result
        except Exception as e:
            print("error: {}".format(e))
            return ""
