import os
from flask_restplus import fields
from werkzeug.datastructures import FileStorage
from werkzeug.exceptions import BadRequest
from maxfw.core import MAX_API, PredictAPI
from core.model import ModelWrapper


# set up parser for audio input data
input_parser = MAX_API.parser()
input_parser.add_argument('audio', type=FileStorage, location='files', required=True,
                          help="signed 16-bit PCM WAV audio file")

predict_response = MAX_API.model('ModelPredictResponse', {
    'status': fields.String(required=True, description='Response status message'),
    'embedding': fields.List(fields.List(fields.Float, required=True, description="Generated embedding"))
})


class ModelPredictAPI(PredictAPI):
    model_wrapper = ModelWrapper()

    @MAX_API.doc('predict')
    @MAX_API.expect(input_parser)
    @MAX_API.marshal_with(predict_response)
    def post(self):
        """Generate audio embedding from input data"""
        result = {'status': 'error'}

        args = input_parser.parse_args()
        audio_data = args['audio'].read()

        # clean up from earlier runs
        if os.path.exists("/audio.wav"):
            os.remove("/audio.wav")

        if '.wav' in str(args['audio']):
            file = open("/audio.wav", "wb")
            file.write(audio_data)
            file.close()
        else:
            e = BadRequest()
            e.data = {'status': 'error', 'message': 'Invalid file type/extension'}
            raise e

        # Getting the predictions
        preds = self.model_wrapper.predict("/audio.wav")

        # Aligning the predictions to the required API format
        result['embedding'] = preds.tolist()
        result['status'] = 'ok'

        os.remove("/audio.wav")

        return result
