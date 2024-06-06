FROM python:3.8

WORKDIR /code

COPY requirements.txt .

RUN pip3.8 install --no-cache-dir -r requirements.txt

COPY . /code/

EXPOSE 8501

CMD ["python3.8", "-m", "streamlit", "run", "streamlit_app_basic.py"]