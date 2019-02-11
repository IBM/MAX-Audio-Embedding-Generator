import tensorflow as tf
from . import vggish_input
from . import vggish_params
from . import vggish_postprocess
from . import vggish_slim
from config import DEFAULT_EMBEDDING_CHECKPOINT, DEFAULT_PCA_PARAMS, MODEL_META_DATA as model_meta
from maxfw.model import MAXModelWrapper


class ModelWrapper(MAXModelWrapper):
    """
    Contains core functions to generate embeddings and classify them.
    Also contains any helper function required.
    """

    MODEL_META_DATA = model_meta

    def __init__(self, embedding_checkpoint=DEFAULT_EMBEDDING_CHECKPOINT, pca_params=DEFAULT_PCA_PARAMS):
        # Initialize the vgg-ish embedding model
        self.graph_embedding = tf.Graph()
        with self.graph_embedding.as_default():
            self.session_embedding = tf.Session()
            vggish_slim.define_vggish_slim(training=False)
            vggish_slim.load_vggish_slim_checkpoint(self.session_embedding, embedding_checkpoint)
            self.features_tensor = self.session_embedding.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
            self.embedding_tensor = self.session_embedding.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)

        # Prepare a postprocessor to munge the vgg-ish model embeddings.
        self.pproc = vggish_postprocess.Postprocessor(pca_params)

    def _generate_embeddings(self, wav_file):
        """
        Generates embeddings as per the Audioset VGG-ish model.
        Post processes embeddings with PCA Quantization
        Input args:
            wav_file   = /path/to/audio/in/wav/format.wav
        Returns:
                Embeddings.
        """
        examples_batch = vggish_input.wavfile_to_examples(wav_file)
        [embedding_batch] = self.session_embedding.run([self.embedding_tensor],
                                                       feed_dict={self.features_tensor: examples_batch})
        return self.pproc.postprocess(embedding_batch)

    def _predict(self, wav_file):
        """
        Driver function that performs all core tasks.
        Input args:
            wav_file = /path/to/audio_file.wav
        Returns:
            embeddings = numpy array of shape (128, length_of_audio_file_seconds).
        """
        embeddings = self._generate_embeddings(wav_file)
        return embeddings
