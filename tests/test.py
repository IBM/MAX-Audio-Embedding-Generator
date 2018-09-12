import pytest
import requests


def test_response():

    model_endpoint = 'http://localhost:5000/model/predict'
    file_path = 'assets/car-horn.wav'

    with open(file_path, 'rb') as file:
        file_form = {'audio': (file_path, file, 'audio/wav')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 200

    response = r.json()

    assert response['status'] == 'ok'

    assert len(response['embedding']) == 4
    assert len(response['embedding'][0]) == 128
    assert len(response['embedding'][1]) == 128
    assert len(response['embedding'][2]) == 128
    assert len(response['embedding'][3]) == 128


if __name__ == '__main__':
    pytest.main([__file__])
