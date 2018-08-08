FROM continuumio/miniconda3

ARG model_bucket=http://max-assets.s3-api.us-geo.objectstorage.softlayer.net/audioset
ARG model_file=assets.tar.gz

WORKDIR /workspace
RUN mkdir assets

RUN wget -nv --show-progress --progress=bar:force:noscroll ${model_bucket}/${model_file} --output-document=/workspace/assets/${model_file}
RUN tar -x -C assets/ -f assets/${model_file} -v && rm assets/${model_file}

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

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

CMD python app.py
