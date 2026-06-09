# Replace this line with your serving base image if needed.
FROM python:3.12-slim

ENV PATH=$HOME/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN sudo mkdir -p /project
COPY . /project/
RUN sudo chown -R $DSUSER:$DSUSER /project

RUN if command -v apt-get >/dev/null 2>&1; then \
        sudo apt-get update \
        && if apt-cache show libdmtx0t64 >/dev/null 2>&1; then \
            sudo apt-get install -y --no-install-recommends libdmtx0t64; \
        else \
            sudo apt-get install -y --no-install-recommends libdmtx0b; \
        fi \
        && sudo apt-get install -y --no-install-recommends openjdk-17-jre-headless curl \
        && sudo rm -rf /var/lib/apt/lists/*; \
    else \
        echo "apt-get not found; assuming native libdmtx is already available in the base image"; \
    fi

RUN pip install -U pip
RUN pip install -U setuptools
RUN pip install -r /project/requirements.txt --no-cache-dir

WORKDIR /project

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8080", "--server.headless", "true"]
