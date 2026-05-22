FROM code.nap.av.it.pt:5050/ai4sme/mobfedlearnsys/mfls-interface:latest

ARG TARGETARCH

VOLUME /app/dataset
VOLUME /app/logs

ENV PYTHONPATH=/app

WORKDIR /app

COPY deployment/requirements.txt /app/requirements_clientML.txt

RUN sed -i 's|http://deb.debian.org/debian|http://archive.debian.org/debian|g' /etc/apt/sources.list && \
	sed -i 's|http://security.debian.org/debian-security|http://archive.debian.org/debian-security|g' /etc/apt/sources.list && \
	sed -i '/buster-updates/d' /etc/apt/sources.list && \
	apt-get -o Acquire::Check-Valid-Until=false update && \
	apt-get install -y --no-install-recommends \
		libsndfile1 \
		ffmpeg \
	&& rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip && \
	if [ "$TARGETARCH" = "arm64" ]; then \
		pip3 install --no-cache-dir --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 torch; \
	else \
		pip3 install --no-cache-dir 'torch>=2.3,<3'; \
	fi && \
	pip3 install --no-cache-dir -r requirements_clientML.txt

COPY cmd /app/

EXPOSE 5000
CMD ["python3", "main.py"]

