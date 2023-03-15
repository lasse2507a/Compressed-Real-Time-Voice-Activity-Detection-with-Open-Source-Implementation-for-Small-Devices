FROM python:3.9

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

COPY ./src ./src

CMD ["python", "./src/main.py"]