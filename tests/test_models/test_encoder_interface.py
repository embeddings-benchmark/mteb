import mteb
from mteb import EncoderProtocol


def test_abs_model_is_encoder():
    model = mteb.get_model("sentence-transformers/all-MiniLM-L6-v2")
    assert isinstance(model, EncoderProtocol)
