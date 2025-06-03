FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
RUN useradd -m -u 1000 user
RUN mkdir -p /home/user/.cache/ace-step/checkpoints && chown -R 1000:1000 /home/user/.cache
RUN --mount=type=cache,target=/home/user/.cache chown 1000:1000 /home/user/.cache
RUN --mount=type=cache,target=/root/.conda conda install -y sox
USER 1000:1000
WORKDIR /app
COPY requirements.txt /app/
RUN --mount=type=cache,target=/home/user/.cache pip install -r /app/requirements.txt
COPY --chown=1000:1000 . /app
RUN --mount=type=cache,target=/home/user/.cache pip install -e .
ENTRYPOINT ["/home/user/.local/bin/acestep"]
CMD ["--server_name", "0.0.0.0"]
