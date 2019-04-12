# Flask settings
DEBUG = False

# Flask-restplus settings
RESTPLUS_MASK_SWAGGER = False

# Application settings

# API metadata
API_TITLE = 'MAX Audio Embedding Generator'
API_DESC = 'Generate embedding vectors from audio files.'
API_VERSION = '1.1.0'


DEFAULT_EMBEDDING_CHECKPOINT = "assets/vggish_model.ckpt"
DEFAULT_PCA_PARAMS = "assets/vggish_pca_params.npz"
