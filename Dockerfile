FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./data /code/data

COPY ./train.py /code/train.py

COPY ./test.py /code/test.py

RUN python /code/train.py

EXPOSE 80

CMD ["uvicorn", "test:app", "--host", "0.0.0.0", "--port", "80"]
