FROM python:3.8-slim-buster

# copy the requirements file into the image
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt
# copy every content from the local file to the image
COPY . /app 
COPY score.py .
COPY app.py . 
# configure the container to run in an executed manner
ENTRYPOINT [ "python" ]
CMD ["app.py", "score.py" ]
