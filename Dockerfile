FROM python:3.8.10-buster
WORKDIR /binarydriller
COPY assets ./assets
COPY analysis ./analysis
COPY *.py ./
COPY pyproject.toml .
COPY poetry.lock .
COPY config.toml config.toml

RUN apt install g++ gcc libc6-dev make
RUN apt install libffi-dev
RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry install

CMD ["poetry", "run", "python", "./main.py" ]
