from flask_restplus import Namespace, Resource, fields
from werkzeug.datastructures import FileStorage

from config import MODEL_META_DATA

import os
import numpy as np

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
        '''Return the metadata associated with the model'''
        return MODEL_META_DATA


predict_response = api.model('ModelPredictResponse', {
    'status': fields.String(required=True, description='Response status message'),
    'result': fields.List(fields.List(fields.Integer))
})

# set up parser for image input data
audio_parser = api.parser()
audio_parser.add_argument('audio', type=FileStorage, location='files', required=True)


@api.route('/predict')
class Predict(Resource):
    @api.doc('predict')
    @api.expect(audio_parser)
    @api.marshal_with(predict_response)
    def post(self):
        '''Generate embeddings from input data'''
        result = {'status': 'error'}

        args = audio_parser.parse_args()
        audio_data = args['audio'].read()

        # clean up from earlier runs
        if os.path.exists("/audio.wav"):
            os.remove("/audio.wav")
        if os.path.exists("/postprocessed_batch.npy"):
            os.remove("/postprocessed_batch.npy")

        file = open("/audio.wav", "wb")
        file.write(audio_data)
        file.close()

        cmd = "python /workspace/core/vggish_inference_demo.py --wav_file /audio.wav" \
              "                                                --checkpoint /workspace/assets/vggish_model.ckpt" \
              "                                                --pca_params /workspace/assets/vggish_pca_params.npz"

        output = os.system(cmd)

        if output != 0:
            result = {'status': 'error'}
            return result

        embeddings = np.load("/postprocessed_batch.npy")

        import json_tricks
        import json

        embeddings_json = json.loads(json_tricks.dumps(embeddings))

        # result['result'] = [{'embeddings': embeddings_json['__ndarray__']}]
        result['result'] = embeddings_json['__ndarray__']
        result['status'] = 'ok'

        os.remove("/audio.wav")
        os.remove("/postprocessed_batch.npy")

        return result
