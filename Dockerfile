FROM python:3.10-slim

# Set the working directory
WORKDIR /code

# Copy requirements.txt to the container
COPY ./requirements.txt /code/requirements.txt

# Install the specified packages
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Download spaCy model and NLTK data during the build process
RUN python -m spacy download en_core_web_sm && \
    mkdir -p /opt/nltk_data && \
    python -m nltk.downloader -d /opt/nltk_data punkt && \
    python -m nltk.downloader -d /opt/nltk_data stopwords

# Set NLTK_DATA environment variable to point to the correct directory
ENV NLTK_DATA=/opt/nltk_data

# Copy the entire application code to the container
COPY ./app /code/app

# Expose port 80
EXPOSE 80

# Set the command to run the FastAPI app with `fastapi run`
CMD ["fastapi", "run", "app/main.py", "--port", "80"]