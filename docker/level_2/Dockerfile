FROM python:3.8

WORKDIR /code

COPY requirements.txt .

RUN pip3.8 install --no-cache-dir -r requirements.txt

COPY . /code/

EXPOSE 5000

CMD ["python3.8", "main.py"]