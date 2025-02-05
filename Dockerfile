FROM python:3.12

# Install pipenv
RUN pip install pipenv==2024.4.0

# Create and set working directory
RUN mkdir -p /home/app
WORKDIR /home/app

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install dependencies from Pipfile
RUN pipenv install --system --deploy

# Copy the rest of the application
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app.py"]
