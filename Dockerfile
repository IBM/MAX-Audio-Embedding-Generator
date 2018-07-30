FROM continuumio/miniconda3

RUN mkdir -p /workspace/assets

RUN wget -nv --show-progress --progress=bar:force:noscroll http://max-assets.s3-api.us-geo.objectstorage.softlayer.net/audioset/assets.tar.gz && mv assets.tar.gz /workspace/assets/
RUN tar -x -C /workspace/assets/ -f /workspace/assets/assets.tar.gz -v && rm /workspace/assets/assets.tar.gz

# Python package versions
ARG numpy_version=1.13.1
ARG tf_version=1.8.0
ARG scipy_version=0.19.1
ARG six_version=1.10.0

RUN pip install --upgrade pip && \
	pip install numpy==${numpy_version} && \
    pip install tensorflow==${tf_version} && \
    pip install scipy==${scipy_version} && \
    pip install resampy && \
    pip install six==${six_version} && \
    pip install flask-restplus && \
    pip install json_tricks

COPY . /workspace

EXPOSE 5000

CMD python workspace/app.py
