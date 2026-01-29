FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY app.py .

# Crear script de limpieza
RUN echo '#!/bin/bash\nfind /tmp/gradio -type f -mmin +60 -delete 2>/dev/null\nfind /tmp/gradio -type d -empty -delete 2>/dev/null' > /usr/local/bin/cleanup.sh \
    && chmod +x /usr/local/bin/cleanup.sh

# Agregar cron job para limpieza cada hora
RUN echo "0 * * * * /usr/local/bin/cleanup.sh" | crontab -

# Exponer el puerto
EXPOSE 7860

# Variables de entorno
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Comando para ejecutar la aplicación con cron
CMD cron && python app.py
