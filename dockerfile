FROM python:3.11-slim

WORKDIR /app

# Copy your application files
COPY tools.py bc3_writer.py /app/

# Install dependencies
# Pin mcp < 2.0.0: mcp 2.0.0 removed streamablehttp_client, which mcpo imports.
RUN pip install "mcpo" "mcp[cli]<2.0.0"

# Set environment variables with defaults
ENV MCP_HOST="0.0.0.0"
ENV MCP_PORT=8000
ENV BLENDER_HOST="host.docker.internal" 
ENV BLENDER_PORT=9876

# Expose the port
EXPOSE ${MCP_PORT}

# Create a startup script that will modify the tools.py file before running
RUN echo '#!/bin/bash\n\
# Replace localhost with the BLENDER_HOST environment variable in tools.py\n\
sed -i "s/host=\"localhost\"/host=\"$BLENDER_HOST\"/g" tools.py\n\
sed -i "s/host='\''localhost'\''/host='\''$BLENDER_HOST'\''/g" tools.py\n\
# Print the modification for debugging\n\
echo "Modified Blender host to: $BLENDER_HOST"\n\
# Run the MCPO server (use the pip-installed mcpo so the mcp<2.0.0 pin applies)\n\
mcpo --host $MCP_HOST --port $MCP_PORT -- python tools.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"]