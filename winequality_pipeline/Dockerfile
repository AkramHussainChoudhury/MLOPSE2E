
FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN ls -la

RUN pip install -r requirements.txt


# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run app.py when the container launches
CMD ["python3", "app.py"]


