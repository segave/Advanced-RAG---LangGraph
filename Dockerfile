FROM python:3.12

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry==1.8.5

# Create and set working directory
RUN mkdir -p /home/app
WORKDIR /home/app

# Copy pyproject.toml and poetry.lock
COPY pyproject.toml poetry.lock ./

# Install dependencies from poetry
RUN poetry lock --no-update && poetry install

# Copy the rest of the application
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
CMD ["poetry", "run", "streamlit", "run", "app.py"]
