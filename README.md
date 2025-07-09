[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/jotaderodriguez-bonsai-mcp-badge.png)](https://mseep.ai/app/jotaderodriguez-bonsai-mcp)


# Bonsai-mcp - Model Context Protocol Integration for IFC through IfcOpenShell and Blender

  

Bonsai-mcp is a fork of [BlenderMCP](https://github.com/ahujasid/blender-mcp) that extends the original functionality with dedicated support for IFC (Industry Foundation Classes) models through Bonsai (previously called BlenderBIM). This integration is a platform to let LLMs read and modify IFC files. 

## Features

-  **IFC-specific functionality**: Query IFC models, analyze spatial structures, and examine building elements

-  **Eleven IFC tools included**: Inspect project info, list entities, examine properties, explore spatial structure, analyze relationships and more

-  **Sequential Thinking**: Includes the sequential thinking tool from [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking) for structured problem solving

-  **Execute Code tool from the original BlenderMCP implementation**: Create and modify objects, apply materials, and execute Python code in Blender

  

## Components


The system consists of two main components:

  

1.  **Blender Addon (`addon.py`)**: A Blender addon that creates a socket server within Blender to receive and execute commands, including IFC-specific operations

2.  **MCP Server (`tools.py`)**: A Python server that implements the Model Context Protocol and connects to the Blender addon

  

## Installation - Through MCP Client Settings

  

### Prerequisites

  

- Blender 4.0 or newer

- Python 3.12 or newer

- uv package manager

- Bonsai BIM addon for Blender (for IFC functionality)

  

**Installing uv:**

  

**Mac:**

```bash

brew  install  uv

```

  

**Windows:**

```bash

powershell  -c  "irm https://astral.sh/uv/install.ps1 | iex"

set  Path=C:\Users\[username]\.local\bin;%Path%

```

  

For other platforms, see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

  

### Clone the repository

  

```bash

git  clone  https://github.com/JotaDeRodriguez/Bonsai_mcp

```

  

### Claude for Desktop Integration

  

Edit your `claude_desktop_config.json` file (Claude > Settings > Developer > Edit Config) to include:

  

```json

{
    "mcpServers": {
        "Bonsai-mcp": {
            "command": "uv",
            "args": [
              "--directory",
              "\\your\\path\\to\\Bonsai_mcp",
              "run",
              "tools.py"
          ]
        }
    }
}

```

## Installation via Docker

The repository comes with a Dockerfile that makes deployment simple and consistent across different environments.
## Quick Start
```bash
# Clone the repository
git clone https://github.com/JotaDeRodriguez/Bonsai_mcp
cd Bonsai_mcp

# Build the Docker image
docker build -t bonsai_mcp .

# Run the container
docker run -p 8000:8000 --name bonsai_mcp bonsai_mcp
```
Once running, the container will expose the MCP tools as REST/OpenAPI APIs at `http://localhost:8000`.

- To verify youtr installation, open your browser and navigate to
- `http://localhost:8000/docs`
-   You'll see the Swagger UI with all available endpoints
-   Test an endpoint by clicking on it, then click "Try it out" and "Execute"

### Connecting to Open WebUI or Other API Clients

To connect this API to Open WebUI:

1.  In Open WebUI, go to Settings > Manage Tool Servers
2.  Add a new connection with:
    -   URL: `http://localhost:8000`
    -   Path to OpenAPI spec: `/openapi.json`
    -   Authentication: None (unless configured otherwise)
    - 
### Environment Variables
The Docker container accepts several environment variables to customize its behavior:
```bash
# Example with custom settings
docker run -p 8000:8000 \
  -e BLENDER_HOST=host.docker.internal \
  -e BLENDER_PORT=9876 \
  -e MCP_HOST=0.0.0.0 \
  -e MCP_PORT=8000 \
  --name bonsai_mcp bonsai_mcp
```
   

## Installing the Blender Addon

1. Download the `addon.py` file from this repo

2. Open Blender

3. Go to Edit > Preferences > Add-ons

4. Click "Install..." and select the `addon.py` file

5. Enable the addon by checking the box next to "Interface: Blender MCP - IFC"

  

## Usage

  

### Starting the Connection

  

1. In Blender, go to the 3D View sidebar (press N if not visible)

2. Find the "Blender MCP - IFC" tab

3. Click "Connect to Claude"

4. Make sure the MCP server is running

  

### Using with Claude

  

Once connected, you'll see a hammer icon in Claude's interface with tools for the Blender MCP IFC integration.

  


## IFC Tools

This repo includes nine IFC-specific tools that enable comprehensive querying and manipulation of IFC models:

**get_ifc_project_info**: Retrieves basic information about the IFC project, including name, description, and counts of different entity types. Example: "What is the basic information about this IFC project?"

**list_ifc_entities**: Lists IFC entities of a specific type (walls, doors, spaces, etc.) with options to limit results and filter by selection. Example: "List all the walls in this IFC model" or "Show me the windows in this building"

**get_ifc_properties**: Retrieves all properties of a specific IFC entity by its GlobalId or from currently selected objects. Example: "What are the properties of this wall with ID 1Dvrgv7Tf5IfTEapMkwDQY?"

**get_ifc_spatial_structure**: Gets the spatial hierarchy of the IFC model (site, building, storeys, spaces). Example: "Show me the spatial structure of this building"

**get_ifc_relationships**: Retrieves all relationships for a specific IFC entity. Example: "What are the relationships of the entrance door?"

**get_selected_ifc_entities**: Gets information about IFC entities corresponding to objects currently selected in the Blender UI. Example: "Tell me about the elements I've selected in Blender"

**get_user_view**: Captures the current Blender viewport as an image, allowing visualization of the model from the user's perspective. Example: "Show me what the user is currently seeing in Blender"

**export_ifc_data**: Exports IFC data to a structured JSON or CSV file, with options to filter by entity type or building level. Example: "Export all wall data to a CSV file"

**place_ifc_object**: Creates and positions an IFC element in the model at specified coordinates with optional rotation. Example: "Place a door at coordinates X:10, Y:5, Z:0 with 90 degrees rotation"

## Execute Blender Code

Legacy feature from the original MCP implementation. Allows Claude to execute arbitrary Python code in Blender. Use with caution.

## Sequential Thinking Tool

This integration includes the Sequential Thinking tool for structured problem-solving and analysis. It facilitates a step-by-step thinking process that can branch, revise, and adapt as understanding deepens - perfect for complex IFC model analysis or planning tasks.

Example: "Use sequential thinking to analyze this building's energy efficiency based on the IFC model"

  

## Example Commands

  

Here are some examples of what you can ask Claude to do with IFC models:

  

- "Analyze this IFC model and tell me how many walls, doors and windows it has"

- "Show me the spatial structure of this building model"

- "List all spaces in this IFC model and their properties"

- "Identify all structural elements in this building"

- "What are the relationships between this wall and other elements?"

- "Use sequential thinking to create a maintenance plan for this building based on the IFC model"

  


## Troubleshooting

-   **Connection issues**: Make sure the Blender addon server is running, and the MCP server is configured in Claude
-   **IFC model not loading**: Verify that you have the Bonsai BIM addon installed and that an IFC file is loaded
-   **Timeout errors**: Try simplifying your requests or breaking them into smaller steps

**Docker:**

-   **"Connection refused" errors**: Make sure Blender is running and the addon is enabled with the server started
-   **CORS issues**: The API has CORS enabled by default for all origins. If you encounter issues, check your client's CORS settings
-   **Performance concerns**: For large IFC models, the API responses might be slower. Consider adjusting timeouts in your client
  

## Technical Details

  

The IFC integration uses the Bonsai BIM module to access ifcopenshell functionality within Blender. The communication follows the same JSON-based protocol over TCP sockets as the original BlenderMCP.

  

## Limitations & Security Considerations

  

- The `execute_blender_code` tool from the original project is still available, allowing running arbitrary Python code in Blender. Use with caution and always save your work.

- Complex IFC models may require breaking down operations into smaller steps.

- IFC query performance depends on model size and complexity.

- Get User View tool returns a base64 encoded image. Please ensure the client supports it.


## Contributions
This MIT licensed repo is open to be forked, modified and used in any way. I'm open to ideas and collaborations, so don't hesitate to get in contact with me for contributions.

  

## Credits

  

- Original BlenderMCP by [Siddharth Ahuja](https://github.com/ahujasid/blender-mcp)

- Sequential Thinking tool from [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking)

- IFC integration built upon the Bonsai BIM addon for Blender

  

## TO DO

Integration and testing with more MCP Clients