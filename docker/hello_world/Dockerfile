FROM python:3.8

WORKDIR /code

COPY index.html .

COPY . /code/

EXPOSE 8000

CMD ["python3.8", "-m", "http.server"]