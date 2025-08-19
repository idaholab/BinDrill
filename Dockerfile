FROM python:3.10-buster
WORKDIR /binarydriller
COPY pyproject.toml .
COPY poetry.lock .
RUN apt install g++ gcc libc6-dev make
RUN apt install libffi-dev
RUN pip install --upgrade pip
RUN pip install poetry
RUN poetry install --no-root
RUN poetry run pip install scikit-learn\<1.2.0
RUN poetry run pip install numpy==1.23.3
COPY config.toml config.toml
COPY assets ./assets
COPY analysis ./analysis
COPY *.py ./
COPY *.sqlite ./

CMD ["poetry", "run", "python", "./main.py" ]
