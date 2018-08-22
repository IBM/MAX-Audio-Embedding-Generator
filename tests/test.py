import pytest
import pycurl
import io
import json


def test_response():
    c = pycurl.Curl()
    b = io.BytesIO()
    c.setopt(pycurl.URL, 'http://localhost:5000/model/predict')
    c.setopt(pycurl.HTTPHEADER, ['Accept:application/json', 'Content-Type: multipart/form-data'])
    c.setopt(pycurl.HTTPPOST, [('audio', (pycurl.FORM_FILE, "assets/car-horn.wav"))])
    c.setopt(pycurl.WRITEFUNCTION, b.write)
    c.perform()
    assert c.getinfo(pycurl.RESPONSE_CODE) == 200
    c.close()

    response = b.getvalue()
    response = json.loads(response)

    assert response['status'] == 'ok'

    assert len(response['predictions'][0]['embedding']) == 4
    assert len(response['predictions'][0]['embedding'][0]) == 128
    assert len(response['predictions'][0]['embedding'][1]) == 128
    assert len(response['predictions'][0]['embedding'][2]) == 128
    assert len(response['predictions'][0]['embedding'][3]) == 128


if __name__ == '__main__':
    pytest.main([__file__])
