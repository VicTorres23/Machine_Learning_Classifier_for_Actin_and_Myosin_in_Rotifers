FROM python:3.12

WORKDIR /workspace

RUN apt-get update && apt-get install -y hmmer && rm -rf /var/lib/apt/lists/*

RUN pip install \
	pandas \
	numpy \
	scikit-learn \
	matplotlib \
	seaborn \
	xgboost \
	joblib

CMD ["bash"]

