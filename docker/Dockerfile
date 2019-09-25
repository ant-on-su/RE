FROM tensorflow/tensorflow:nightly-gpu-py3-jupyter

# Python dependencies
ADD requirements.txt /
RUN pip3 install --upgrade pip
RUN pip3 install -r /requirements.txt

RUN echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.port = 8080" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py

WORKDIR /root
EXPOSE 8080
