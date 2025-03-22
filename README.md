# Bonsai-mcp - Blender Model Context Protocol Integration for IFC through IfcOpenShell

Bonsai-mcp is a fork of [BlenderMCP](https://github.com/ahujasid/blender-mcp) that extends the original functionality with dedicated support for IFC (Industry Foundation Classes) models through Bonsai. This integration is a quick proof of concept aimed at exemplifying the capabilites of connecting Claude, or any LLM (though this was only tested using the Claude Desktop Client), to Blender in order to execute IfcOpenShell commands.

## Features

- **IFC-specific functionality**: Query IFC models, analyze spatial structures, and examine building elements
- **Five powerful IFC tools**: Inspect project info, list entities, examine properties, explore spatial structure, and analyze relationships
- **Sequential Thinking**: Includes the sequential thinking tool from [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking) for structured problem solving
- **Execute Code tool from the original BlenderMCP implementation**: Create and modify objects, apply materials, and execute Python code in Blender
- **Tested with standard models**: Verified working with the default ifcopenshell house model ([AC20-FZK-Haus.ifc](https://www.ifcwiki.org/images/e/e3/AC20-FZK-Haus.ifc))

## Components

The system consists of two main components:

1. **Blender Addon (`addon.py`)**: A Blender addon that creates a socket server within Blender to receive and execute commands, including IFC-specific operations
2. **MCP Server (`blender_mcp_tools.py`)**: A Python server that implements the Model Context Protocol and connects to the Blender addon

## Installation

### Prerequisites

- Blender 3.0 or newer
- Python 3.10 or newer
- uv package manager
- Bonsai BIM addon for Blender (for IFC functionality)

**Installing uv:**

**Mac:**
```bash
brew install uv
```

**Windows:**
```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | iex" 
set Path=C:\Users\[username]\.local\bin;%Path%
```

For other platforms, see the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

### Clone the repository

```bash
git clone https://github.com/JotaDeRodriguez/Bonsai_mcp
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

### Installing the Blender Addon

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

This fork adds five powerful IFC-specific tools:

### 1. get_ifc_project_info

Get basic information about the IFC project, including name, description, and counts of different entity types.

Example: "What is the basic information about this IFC project?"

### 2. list_ifc_entities

List IFC entities of a specific type (walls, doors, spaces, etc.).

Example: "List all the walls in this IFC model" or "Show me the windows in this building"

### 3. get_ifc_properties

Get all properties of a specific IFC entity by its GlobalId.

Example: "What are the properties of this wall with ID 1Dvrgv7Tf5IfTEapMkwDQY?"

### 4. get_ifc_spatial_structure

Get the spatial hierarchy of the IFC model (site, building, storeys, spaces).

Example: "Show me the spatial structure of this building"

### 5. get_ifc_relationships

Get all relationships for a specific IFC entity.

Example: "What are the relationships of the entrance door?"

## Excecute Blender Code

Legacy feature from the original MCP implementation. Allows Claude to execute arbitrary Python code in Blender. Use with caution.

## Sequential Thinking Tool

This integration also includes the Sequential Thinking tool, which facilitates a detailed, step-by-step thinking process for problem-solving and analysis.


### Tool parameters:

- `thought` (string): The current thinking step
- `nextThoughtNeeded` (boolean): Whether another thought step is needed
- `thoughtNumber` (integer): Current thought number
- `totalThoughts` (integer): Estimated total thoughts needed
- `isRevision` (boolean, optional): Whether this revises previous thinking
- `revisesThought` (integer, optional): Which thought is being reconsidered
- `branchFromThought` (integer, optional): Branching point thought number
- `branchId` (string, optional): Branch identifier
- `needsMoreThoughts` (boolean, optional): If more thoughts are needed

Example: "Use sequential thinking to analyze this building's energy efficiency"

## Example Commands

Here are some examples of what you can ask Claude to do with IFC models:

- "Analyze this IFC model and tell me how many walls, doors and windows it has"
- "Show me the spatial structure of this building model"
- "List all spaces in this IFC model and their properties"
- "Identify all structural elements in this building"
- "What are the relationships between this wall and other elements?"
- "Use sequential thinking to create a maintenance plan for this building based on the IFC model"

## Troubleshooting

- **Connection issues**: Make sure the Blender addon server is running, and the MCP server is configured in Claude
- **IFC model not loading**: Verify that you have the Bonsai BIM addon installed and that an IFC file is loaded
- **Timeout errors**: Try simplifying your requests or breaking them into smaller steps

## Technical Details

The IFC integration uses the Bonsai BIM module to access ifcopenshell functionality within Blender. The communication follows the same JSON-based protocol over TCP sockets as the original BlenderMCP.

## Limitations & Security Considerations

- The `execute_blender_code` tool from the original project is still available, allowing running arbitrary Python code in Blender. Use with caution and always save your work.
- Complex IFC models may require breaking down operations into smaller steps.
- IFC query performance depends on model size and complexity.

## Credits

- Original BlenderMCP by [Siddharth Ahuja](https://github.com/ahujasid/blender-mcp)
- Sequential Thinking tool from [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking)
- IFC integration built upon the Bonsai BIM addon for Blender

## TODO
Cursor implementation