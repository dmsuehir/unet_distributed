FROM tensorflow/tensorflow:1.4.0

ADD requirements.txt /src/deps/requirements.txt
RUN pip install -r /src/deps/requirements.txt
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev

ADD test_dist.py /opt/test_dist.py
ADD settings_dist.py /opt/settings_dist.py
ADD preprocess.py /opt/preprocess.py
ADD model.py /opt/model.py 
ADD sanity_check_trained_model.py /opt/sanity_check_trained_model.py
ADD data.py /opt/data.py

RUN chmod +x /opt/test_dist.py
RUN chmod +x /opt/settings_dist.py
RUN chmod +x /opt/preprocess.py
RUN chmod +x /opt/model.py
RUN chmod +x /opt/sanity_check_trained_model.py
RUN chmod +x /opt/data.py

ENTRYPOINT ["python", "/opt/test_dist.py", "--is_sync", "0"]
