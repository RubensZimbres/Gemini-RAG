FROM python:3.10

EXPOSE 8080
ENV PORT 8080

# Create appuser with specified UID and GID --- AVOID ROOT USER  FOR SECURITY REASONS
RUN groupadd -g 900 appuser && \
    useradd -r -u 900 -g appuser appuser

# Set the working directory before switching to appuser
WORKDIR /home

# Copy files and set ownership
COPY . /home
RUN chown -R appuser:appuser /home

# Switch to appuser before installing dependencies
USER appuser

# Install dependencies
RUN pip install -r /home/requirements.txt
RUN pip install langchain

# Command to run when the container starts
CMD python3 /home/predict.py
