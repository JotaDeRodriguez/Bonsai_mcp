# blender_mcp_server.py
from mcp.server.fastmcp import FastMCP, Context, Image
import socket
import json
import asyncio
import logging
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any, List, TypedDict
import os
from pathlib import Path
import base64
from urllib.parse import urlparse
from typing import Optional
import sys


# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BlenderMCPServer")

@dataclass
class BlenderConnection:
    host: str
    port: int
    sock: socket.socket = None  # Changed from 'socket' to 'sock' to avoid naming conflict
    
    def connect(self) -> bool:
        """Connect to the Blender addon socket server"""
        if self.sock:
            return True
            
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.host, self.port))
            logger.info(f"Connected to Blender at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Blender: {str(e)}")
            self.sock = None
            return False
    
    def disconnect(self):
        """Disconnect from the Blender addon"""
        if self.sock:
            try:
                self.sock.close()
            except Exception as e:
                logger.error(f"Error disconnecting from Blender: {str(e)}")
            finally:
                self.sock = None

    def receive_full_response(self, sock, buffer_size=8192):
        """Receive the complete response, potentially in multiple chunks"""
        chunks = []
        # Use a consistent timeout value that matches the addon's timeout
        sock.settimeout(15.0)  # Match the addon's timeout
        
        try:
            while True:
                try:
                    chunk = sock.recv(buffer_size)
                    if not chunk:
                        # If we get an empty chunk, the connection might be closed
                        if not chunks:  # If we haven't received anything yet, this is an error
                            raise Exception("Connection closed before receiving any data")
                        break
                    
                    chunks.append(chunk)
                    
                    # Check if we've received a complete JSON object
                    try:
                        data = b''.join(chunks)
                        json.loads(data.decode('utf-8'))
                        # If we get here, it parsed successfully
                        logger.info(f"Received complete response ({len(data)} bytes)")
                        return data
                    except json.JSONDecodeError:
                        # Incomplete JSON, continue receiving
                        continue
                except socket.timeout:
                    # If we hit a timeout during receiving, break the loop and try to use what we have
                    logger.warning("Socket timeout during chunked receive")
                    break
                except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                    logger.error(f"Socket connection error during receive: {str(e)}")
                    raise  # Re-raise to be handled by the caller
        except socket.timeout:
            logger.warning("Socket timeout during chunked receive")
        except Exception as e:
            logger.error(f"Error during receive: {str(e)}")
            raise
            
        # If we get here, we either timed out or broke out of the loop
        # Try to use what we have
        if chunks:
            data = b''.join(chunks)
            logger.info(f"Returning data after receive completion ({len(data)} bytes)")
            try:
                # Try to parse what we have
                json.loads(data.decode('utf-8'))
                return data
            except json.JSONDecodeError:
                # If we can't parse it, it's incomplete
                raise Exception("Incomplete JSON response received")
        else:
            raise Exception("No data received")

    def send_command(self, command_type: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Send a command to Blender and return the response"""
        if not self.sock and not self.connect():
            raise ConnectionError("Not connected to Blender")
        
        command = {
            "type": command_type,
            "params": params or {}
        }
        
        try:
            # Log the command being sent
            logger.info(f"Sending command: {command_type} with params: {params}")
            
            # Send the command
            self.sock.sendall(json.dumps(command).encode('utf-8'))
            logger.info(f"Command sent, waiting for response...")
            
            # Set a timeout for receiving - use the same timeout as in receive_full_response
            self.sock.settimeout(15.0)  # Match the addon's timeout
            
            # Receive the response using the improved receive_full_response method
            response_data = self.receive_full_response(self.sock)
            logger.info(f"Received {len(response_data)} bytes of data")
            
            response = json.loads(response_data.decode('utf-8'))
            logger.info(f"Response parsed, status: {response.get('status', 'unknown')}")
            
            if response.get("status") == "error":
                logger.error(f"Blender error: {response.get('message')}")
                raise Exception(response.get("message", "Unknown error from Blender"))
            
            return response.get("result", {})
        except socket.timeout:
            logger.error("Socket timeout while waiting for response from Blender")
            # Don't try to reconnect here - let the get_blender_connection handle reconnection
            # Just invalidate the current socket so it will be recreated next time
            self.sock = None
            raise Exception("Timeout waiting for Blender response - try simplifying your request")
        except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
            logger.error(f"Socket connection error: {str(e)}")
            self.sock = None
            raise Exception(f"Connection to Blender lost: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from Blender: {str(e)}")
            # Try to log what was received
            if 'response_data' in locals() and response_data:
                logger.error(f"Raw response (first 200 bytes): {response_data[:200]}")
            raise Exception(f"Invalid response from Blender: {str(e)}")
        except Exception as e:
            logger.error(f"Error communicating with Blender: {str(e)}")
            # Don't try to reconnect here - let the get_blender_connection handle reconnection
            self.sock = None
            raise Exception(f"Communication error with Blender: {str(e)}")

@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Manage server startup and shutdown lifecycle"""
    # We don't need to create a connection here since we're using the global connection
    # for resources and tools
    
    try:
        # Just log that we're starting up
        logger.info("BlenderMCP server starting up")
        
        # Try to connect to Blender on startup to verify it's available
        try:
            # This will initialize the global connection if needed
            blender = get_blender_connection()
            logger.info("Successfully connected to Blender on startup")
        except Exception as e:
            logger.warning(f"Could not connect to Blender on startup: {str(e)}")
            logger.warning("Make sure the Blender addon is running before using Blender resources or tools")
        
        # Return an empty context - we're using the global connection
        yield {}
    finally:
        # Clean up the global connection on shutdown
        global _blender_connection
        if _blender_connection:
            logger.info("Disconnecting from Blender on shutdown")
            _blender_connection.disconnect()
            _blender_connection = None
        logger.info("BlenderMCP server shut down")

# Create the MCP server with lifespan support
mcp = FastMCP(
    "Bonsai MCP",
    description="IFC manipulation through Blender and MCP",
    lifespan=server_lifespan
)

# Resource endpoints

# Global connection for resources (since resources can't access context)
_blender_connection = None

def get_blender_connection():
    """Get or create a persistent Blender connection"""
    global _blender_connection
    
    # If we have an existing connection, check if it's still valid
    if _blender_connection is not None:
        try:
            # Simple ping to check if connection is still alive
            _blender_connection.send_command("get_ifc_project_info")
            return _blender_connection
        except Exception as e:
            # Connection is dead, close it and create a new one
            logger.warning(f"Existing connection is no longer valid: {str(e)}")
            try:
                _blender_connection.disconnect()
            except:
                pass
            _blender_connection = None
    
    # Create a new connection if needed
    if _blender_connection is None:
        _blender_connection = BlenderConnection(host="localhost", port=9876)
        if not _blender_connection.connect():
            logger.error("Failed to connect to Blender")
            _blender_connection = None
            raise Exception("Could not connect to Blender. Make sure the Blender addon is running.")
        logger.info("Created new persistent connection to Blender")
    
    return _blender_connection


### Sequencial Thinking Server: 

# Sequential Thinking Tool
class ThoughtData(TypedDict, total=False):
    thought: str
    thoughtNumber: int
    totalThoughts: int
    nextThoughtNeeded: bool
    isRevision: Optional[bool]
    revisesThought: Optional[int]
    branchFromThought: Optional[int]
    branchId: Optional[str]
    needsMoreThoughts: Optional[bool]


class SequentialThinkingServer:
    def __init__(self):
        self.thought_history = []
        self.branches = {}
    
    def validate_thought_data(self, data: Dict[str, Any]) -> ThoughtData:
        if not isinstance(data.get('thought'), str):
            raise ValueError('Invalid thought: must be a string')
        if not isinstance(data.get('thoughtNumber'), int):
            raise ValueError('Invalid thoughtNumber: must be a number')
        if not isinstance(data.get('totalThoughts'), int):
            raise ValueError('Invalid totalThoughts: must be a number')
        if not isinstance(data.get('nextThoughtNeeded'), bool):
            raise ValueError('Invalid nextThoughtNeeded: must be a boolean')
        
        return {
            'thought': data['thought'],
            'thoughtNumber': data['thoughtNumber'],
            'totalThoughts': data['totalThoughts'],
            'nextThoughtNeeded': data['nextThoughtNeeded'],
            'isRevision': data.get('isRevision'),
            'revisesThought': data.get('revisesThought'),
            'branchFromThought': data.get('branchFromThought'),
            'branchId': data.get('branchId'),
            'needsMoreThoughts': data.get('needsMoreThoughts')
        }
    
    def format_thought(self, thought_data: ThoughtData) -> str:
        """Format a thought with colored borders and context"""
        thought_num = thought_data['thoughtNumber']
        total = thought_data['totalThoughts']
        thought = thought_data['thought']
        is_revision = thought_data.get('isRevision', False)
        revises = thought_data.get('revisesThought')
        branch_from = thought_data.get('branchFromThought')
        branch_id = thought_data.get('branchId')
        
        # Create appropriate prefix and context
        if is_revision:
            prefix = "🔄 Revision"
            context = f" (revising thought {revises})"
        elif branch_from:
            prefix = "🌿 Branch"
            context = f" (from thought {branch_from}, ID: {branch_id})"
        else:
            prefix = "💭 Thought"
            context = ""
        
        header = f"{prefix} {thought_num}/{total}{context}"
        border_len = max(len(header), len(thought)) + 4
        border = "─" * border_len
        
        # Build the formatted output
        output = f"\n┌{border}┐\n"
        output += f"│ {header.ljust(border_len)} │\n"
        output += f"├{border}┤\n"
        output += f"│ {thought.ljust(border_len)} │\n"
        output += f"└{border}┘"
        
        return output
    
    def process_thought(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a thought and return the response"""
        try:
            validated_input = self.validate_thought_data(input_data)
            
            if validated_input['thoughtNumber'] > validated_input['totalThoughts']:
                validated_input['totalThoughts'] = validated_input['thoughtNumber']
            
            self.thought_history.append(validated_input)
            
            # Track branches if applicable
            if validated_input.get('branchFromThought') and validated_input.get('branchId'):
                branch_id = validated_input['branchId']
                if branch_id not in self.branches:
                    self.branches[branch_id] = []
                self.branches[branch_id].append(validated_input)
            
            # Format and log the thought
            formatted_thought = self.format_thought(validated_input)
            print(formatted_thought, file=sys.stderr)
            
            # Return response
            return {
                'thoughtNumber': validated_input['thoughtNumber'],
                'totalThoughts': validated_input['totalThoughts'],
                'nextThoughtNeeded': validated_input['nextThoughtNeeded'],
                'branches': list(self.branches.keys()),
                'thoughtHistoryLength': len(self.thought_history)
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'status': 'failed'
            }

# Create a single instance of the sequential thinking server
thinking_server = SequentialThinkingServer()

# -------------------------------
# MCP TOOLS
# -------------------------------


@mcp.tool()
def execute_blender_code(ctx: Context, code: str) -> str:
    """
    Execute arbitrary Python code in Blender.
    
    Parameters:
    - code: The Python code to execute
    """
    try:
        # Get the global connection
        blender = get_blender_connection()
        
        result = blender.send_command("execute_code", {"code": code})
        return f"Code executed successfully: {result.get('result', '')}"
    except Exception as e:
        logger.error(f"Error executing code: {str(e)}")
        return f"Error executing code: {str(e)}"
    

### IFC Tools
@mcp.tool()
def get_ifc_project_info() -> str:
    """
    Get basic information about the IFC project, including name, description, 
    and counts of different entity types.
    
    Returns:
        A JSON-formatted string with project information
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_ifc_project_info")
        
        # Return the formatted JSON of the results
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting IFC project info: {str(e)}")
        return f"Error getting IFC project info: {str(e)}"

@mcp.tool()
def get_selected_ifc_entities() -> str:
    """
    Get IFC entities corresponding to the currently selected objects in Blender.
    This allows working specifically with objects the user has manually selected in the Blender UI.
    
    Returns:
        A JSON-formatted string with information about the selected IFC entities
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_selected_ifc_entities")
        
        # Return the formatted JSON of the results
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting selected IFC entities: {str(e)}")
        return f"Error getting selected IFC entities: {str(e)}"

# Modify the existing list_ifc_entities function to accept a selected_only parameter
@mcp.tool()
def list_ifc_entities(entity_type: str | None = None, limit: int = 50, selected_only: bool = False) -> str:
    """
    List IFC entities of a specific type. Can be filtered to only include objects
    currently selected in the Blender UI.
    
    Args:
        entity_type: Type of IFC entity to list (e.g., "IfcWall")
        limit: Maximum number of entities to return
        selected_only: If True, only return information about selected objects
    
    Returns:
        A JSON-formatted string listing the specified entities
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("list_ifc_entities", {
            "entity_type": entity_type,
            "limit": limit,
            "selected_only": selected_only
        })
        
        # Return the formatted JSON of the results
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error listing IFC entities: {str(e)}")
        return f"Error listing IFC entities: {str(e)}"

# Modify the existing get_ifc_properties function to accept a selected_only parameter
@mcp.tool()
def get_ifc_properties(global_id: str | None = None, selected_only: bool = False) -> str:
    """
    Get properties of IFC entities. Can be used to get properties of a specific entity by GlobalId,
    or to get properties of all currently selected objects in Blender.
    
    Args:
        global_id: GlobalId of a specific IFC entity (optional if selected_only is True)
        selected_only: If True, return properties for all selected objects instead of a specific entity
    
    Returns:
        A JSON-formatted string with entity information and properties
    """
    try:
        blender = get_blender_connection()
        
        # Validate parameters
        if not global_id and not selected_only:
            return json.dumps({"error": "Either global_id or selected_only must be specified"}, indent=2)
        
        result = blender.send_command("get_ifc_properties", {
            "global_id": global_id,
            "selected_only": selected_only
        })
        
        # Return the formatted JSON of the results
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting IFC properties: {str(e)}")
        return f"Error getting IFC properties: {str(e)}"
    
    
@mcp.tool()
def get_ifc_spatial_structure() -> str:
    """
    Get the spatial structure of the IFC model (site, building, storey, space hierarchy).
    
    Returns:
        A JSON-formatted string representing the hierarchical structure of the IFC model
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_ifc_spatial_structure")
        
        # Return the formatted JSON of the results
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting IFC spatial structure: {str(e)}")
        return f"Error getting IFC spatial structure: {str(e)}"

@mcp.tool()
def get_ifc_relationships(global_id: str) -> str:
    """
    Get all relationships for a specific IFC entity.
    
    Args:
        global_id: GlobalId of the IFC entity
    
    Returns:
        A JSON-formatted string with all relationships the entity participates in
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_ifc_relationships", {
            "global_id": global_id
        })
        
        # Return the formatted JSON of the results
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting IFC relationships: {str(e)}")
        return f"Error getting IFC relationships: {str(e)}"
    

@mcp.tool()
def export_ifc_data(
    entity_type: str | None = None, 
    level_name: str | None = None, 
    output_format: str = "csv",
    ctx: Context | None = None
) -> str:
    """
    Export IFC data to a file in JSON or CSV format.
    
    This tool extracts IFC data and creates a structured export file. You can filter
    by entity type and/or building level, and choose the output format.
    
    Args:
        entity_type: Type of IFC entity to export (e.g., "IfcWall") - leave empty for all entities
        level_name: Name of the building level to filter by (e.g., "Level 1") - leave empty for all levels
        output_format: "json" or "csv" format for the output file
        
    Returns:
        Confirmation message with the export file path or an error message
    """
    try:
        # Get Blender connection
        blender = get_blender_connection()

        # Validate output format
        if output_format not in ["json", "csv"]:
            return "Error: output_format must be 'json' or 'csv'"

        # Execute the export code in Blender
        result = blender.send_command("export_ifc_data", {
            "entity_type": entity_type,
            "level_name": level_name,
            "output_format": output_format
        })
        
        # Check for errors from Blender
        if isinstance(result, dict) and "error" in result:
            return f"Error: {result['error']}"
        
        # Return the result with export summary
        # return result
        return json.dumps(result, indent=2)
    
    except Exception as e:
        logger.error(f"Error exporting IFC data: {str(e)}")
        return f"Error exporting IFC data: {str(e)}"
    

@mcp.tool()
def place_ifc_object(
    type_name: str, 
    x: float, 
    y: float, 
    z: float, 
    rotation: float = 0.0,
    ctx: Context| None = None
) -> str:
    """
    Place an IFC object at a specified location with optional rotation.
    
    This tool allows you to create and position IFC elements in the model.
    The object is placed using the specified IFC type and positioned
    at the given coordinates with optional rotation around the Z axis.
    
    Args:
        type_name: Name of the IFC element type to place (must exist in the model)
        x: X-coordinate in model space
        y: Y-coordinate in model space
        z: Z-coordinate in model space
        rotation: Rotation angle in degrees around the Z axis (default: 0)
        
    Returns:
        A message with the result of the placement operation
    """
    try:
        # Get Blender connection
        blender = get_blender_connection()
        
        # Send command to place the object
        result = blender.send_command("place_ifc_object", {
            "type_name": type_name,
            "location": [x, y, z],
            "rotation": rotation
        })
        
        # Check for errors
        if isinstance(result, dict) and "error" in result:
            return f"Error placing object: {result['error']}"
        
        # Format success message
        if isinstance(result, dict) and result.get("success"):
            return (f"Successfully placed '{type_name}' object at ({x}, {y}, {z}) "
                   f"with {rotation}° rotation.\nObject name: {result.get('blender_name')}, "
                   f"Global ID: {result.get('global_id')}")
        
        # Return the raw result as string if it's not a success or error dict
        return f"Placement result: {json.dumps(result, indent=2)}"
    
    except Exception as e:
        logger.error(f"Error placing IFC object: {str(e)}")
        return f"Error placing IFC object: {str(e)}"
    
    
@mcp.tool()
def get_user_view() -> Image:
    """
    Capture and return the current Blender viewport as an image.
    Shows what the user is currently seeing in Blender.

    Focus mostly on the 3D viewport. Use the UI to assist in your understanding of the scene but only refer to it if specifically prompted.
    
    Args:
        max_dimension: Maximum dimension (width or height) in pixels for the returned image
        compression_quality: Image compression quality (1-100, higher is better quality but larger)
    
    Returns:
        An image of the current Blender viewport
    """
    max_dimension = 800
    compression_quality = 85

    # Use PIL to compress the image
    from PIL import Image as PILImage
    import io

    try:
        # Get the global connection
        blender = get_blender_connection()
        
        # Request current view
        result = blender.send_command("get_current_view")
        
        if "error" in result:
            # logger.error(f"Error getting view from Blender: {result.get('error')}")
            raise Exception(f"Error getting current view: {result.get('error')}")
        
        # Extract image information
        if "data" not in result or "width" not in result or "height" not in result:
            # logger.error("Incomplete image data returned from Blender")
            raise Exception("Incomplete image data returned from Blender")
        
        # Decode the base64 image data
        image_data = base64.b64decode(result["data"])
        original_width = result["width"]
        original_height = result["height"]
        original_format = result.get("format", "png")
        
        # Compression is only needed if the image is large
        if original_width > 800 or original_height > 800 or len(image_data) > 1000000:
            # logger.info(f"Compressing image (original size: {len(image_data)} bytes)")
            
            # Open image from binary data
            img = PILImage.open(io.BytesIO(image_data))
            
            # Resize if needed
            if original_width > max_dimension or original_height > max_dimension:
                # Calculate new dimensions maintaining aspect ratio
                if original_width > original_height:
                    new_width = max_dimension
                    new_height = int(original_height * (max_dimension / original_width))
                else:
                    new_height = max_dimension
                    new_width = int(original_width * (max_dimension / original_height))
                
                # Resize using high-quality resampling
                img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS)
            
            # Convert to RGB if needed
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # Save as JPEG with compression
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=compression_quality, optimize=True)
            compressed_data = output.getvalue()

            # logger.info(f"Image compressed from {len(image_data)} to {len(compressed_data)} bytes")
            
            # Return compressed image
            return Image(data=compressed_data, format="jpeg")
        else:
            # Image is small enough, return as-is
            return Image(data=image_data, format=original_format)
            
    except Exception as e:
        # logger.error(f"Error processing viewport image: {str(e)}")
        raise Exception(f"Error processing viewport image: {str(e)}")


@mcp.tool()
def get_ifc_quantities() -> str:
    """
    Extract and get basic qtos about the IFC project.
    
    Returns:
        A JSON-formatted string with project quantities information
    """
    try:
        blender = get_blender_connection()
        result = blender.send_command("get_ifc_quantities")
        
        # Return the formatted JSON of the results
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error getting IFC project quantities: {str(e)}")
        return f"Error getting IFC project quantities: {str(e)}"


# WIP, not ready to be implemented:  
# @mcp.tool()
# def create_plan_view(height_offset: float = 0.5, view_type: str = "top", 
#                     resolution_x: int = 400, resolution_y: int = 400,
#                     output_path: str = None) -> Image:
#     """
#     Create a plan view (top-down view) at the specified height above the first building story.
    
#     Args:
#         height_offset: Height in meters above the building story (default 0.5m)
#         view_type: Type of view - "top", "front", "right", "left" (note: only "top" is fully implemented)
#         resolution_x: Horizontal resolution of the render in pixels - Keep it small, max 800 x 800, recomended 400 x 400
#         resolution_y: Vertical resolution of the render in pixels
#         output_path: Optional path to save the rendered image
    
#     Returns:
#         A rendered image showing the plan view of the model
#     """
#     try:
#         # Get the global connection
#         blender = get_blender_connection()
        
#         # Request an orthographic render
#         result = blender.send_command("create_orthographic_render", {
#             "view_type": view_type,
#             "height_offset": height_offset,
#             "resolution_x": resolution_x,
#             "resolution_y": resolution_y,
#             "output_path": output_path  # Can be None to use a temporary file
#         })
        
#         if "error" in result:
#             raise Exception(f"Error creating plan view: {result.get('error', 'Unknown error')}")
        
#         if "data" not in result:
#             raise Exception("No image data returned from Blender")
        
#         # Decode the base64 image data
#         image_data = base64.b64decode(result["data"])
        
#         # Return as an Image object
#         return Image(data=image_data, format="png")
#     except Exception as e:
#         logger.error(f"Error creating plan view: {str(e)}")
#         raise Exception(f"Error creating plan view: {str(e)}")


@mcp.tool()
def export_drawing_png(
    height_offset: float = 0.5,
    view_type: str = "top",
    resolution_x: int = 1920,
    resolution_y: int = 1080,
    storey_name: str | None = None,
    output_path: str | None = None
) -> dict:
    """Export drawings as PNG images with custom resolution.
    
    Creates a drawing, with the view type specified, of the IFC building at the specified 
    height above the floor level. Supports custom resolution for high-quality architectural drawings.
    
    Args:
        height_offset: Height in meters above the storey level for the camera position (default 0.5m)
        view_type: Type of view - "top" for plan view, "front", "right" and "left" for elevation views, and "isometric" for 3D view
        resolution_x: Horizontal resolution in pixels (default 1920, max recommended 4096)
        resolution_y: Vertical resolution in pixels (default 1080, max recommended 4096)
        storey_name: Specific storey name to add to the file name (if None, prints default in the file name)
        output_path: Optional file path to save the PNG (if None, returns as base64 image)
    
    Returns:
        metadata and the path of the file image of the drawing at the specified resolution
    """
    try:
        # Validate resolution limits for performance
        if resolution_x > 4096 or resolution_y > 4096:
            raise Exception("Resolution too high. Maximum recommended: 4096x4096 pixels")
        
        if resolution_x < 100 or resolution_y < 100:
            raise Exception("Resolution too low. Minimum: 100x100 pixels")
        
        # Get the global connection
        blender = get_blender_connection()
        
        # Request drawing render
        result = blender.send_command("export_drawing_png", {
            "view_type": view_type,
            "height_offset": height_offset,
            "resolution_x": resolution_x,
            "resolution_y": resolution_y,
            "storey_name": storey_name,
            "output_path": output_path
        })
        
        if "error" in result:
            raise Exception(f"Error creating {view_type} drawing: {result.get('error', 'Unknown error')}")
        
        if "data" not in result:
            raise Exception("No image data returned from Blender")
        
        # Decode the base64 image data
        image_data = base64.b64decode(result["data"])
        
        # Ensure output path exists
        if not output_path:
            os.makedirs("./exports/drawings", exist_ok=True)
            # Generate filename based on view type
            view_name = {
                "top": "plan_view",
                "front": "front_elevation", 
                "right": "right_elevation",
                "left": "left_elevation",
                "isometric": "isometric_view"
            }.get(view_type, view_type)
            filename = f"{view_name}_{storey_name or 'default'}.png"
            output_path = os.path.join("./exports/drawings", filename)
        
        # Save to file
        with open(output_path, "wb") as f:
            f.write(image_data)
        
        # Return only metadata
        return {
            "status": "success",
            "file_path": os.path.abspath(output_path),
            # Opcional: si tienes un servidor de archivos, podrías devolver también una URL
            # "url": f"http://localhost:8000/files/{filename}"
        }
        
    except Exception as e:
        logger.error(f"Error exporting drawing: {str(e)}")
        return { "status": "error", "message": str(e) }

@mcp.tool()
def sequentialthinking(
    thought: str,
    thoughtNumber: int,
    totalThoughts: int,
    nextThoughtNeeded: bool,
    isRevision: Optional[bool] = None,
    revisesThought: Optional[int] = None,
    branchFromThought: Optional[int] = None,
    branchId: Optional[str] = None,
    needsMoreThoughts: Optional[bool] = None
) -> str:
    """A detailed tool for dynamic and reflective problem-solving through thoughts.
    
    This tool helps analyze problems through a flexible thinking process that can adapt and evolve.
    Each thought can build on, question, or revise previous insights as understanding deepens.
    
    When to use this tool:
    - Breaking down complex problems into steps
    - Planning and design with room for revision
    - Analysis that might need course correction
    - Problems where the full scope might not be clear initially
    - Problems that require a multi-step solution
    - Tasks that need to maintain context over multiple steps
    - Situations where irrelevant information needs to be filtered out
    
    Args:
        thought: Your current thinking step
        thoughtNumber: Current number in sequence (can go beyond initial total if needed)
        totalThoughts: Current estimate of thoughts needed (can be adjusted up/down)
        nextThoughtNeeded: Whether another thought step is needed
        isRevision: Whether this revises previous thinking
        revisesThought: Which thought is being reconsidered
        branchFromThought: Branching point thought number
        branchId: Branch identifier
        needsMoreThoughts: If more thoughts are needed
    """
    input_data = {
        'thought': thought,
        'thoughtNumber': thoughtNumber,
        'totalThoughts': totalThoughts,
        'nextThoughtNeeded': nextThoughtNeeded
    }
    
    # Add optional parameters if provided
    if isRevision is not None:
        input_data['isRevision'] = isRevision
    if revisesThought is not None:
        input_data['revisesThought'] = revisesThought
    if branchFromThought is not None:
        input_data['branchFromThought'] = branchFromThought
    if branchId is not None:
        input_data['branchId'] = branchId
    if needsMoreThoughts is not None:
        input_data['needsMoreThoughts'] = needsMoreThoughts
    
    response = thinking_server.process_thought(input_data)
    return json.dumps(response, indent=2)


# -------------------------------
# MCP RESOURCES
# -------------------------------

# Base path of the resource files
BASE_PATH = Path("./resources")

@mcp.resource("file://table_of_contents.md")
def formulas_rp() -> str:
    """Leer el contenido del archivo table_of_contents.md"""
    file_path = BASE_PATH / "table_of_contents.md"
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: No se encontró el archivo {file_path}"
    except Exception as e:
        return f"Error al leer el archivo: {str(e)}"


# -------------------------------
# MCP PROMPTS
# -------------------------------

@mcp.prompt("Technical_building_report")
def technical_building_report(project_name: str, project_location: str, language: str = "english") -> str:
    """
    Generate a comprehensive technical building report based on an IFC model loaded in Blender.
    
    Args:
        project_name: Name of the project/building
        project_location: Building location (city, address)
        language: Report language - "english", "spanish", "french", "german", "italian", "portuguese"
    
    Returns:
        Structured technical report following basic project standards in the selected language.
    """
    
    # Language-specific instructions
    language_instructions = {
        "english": {
            "role": "You are a technical architect specialized in creating technical reports for basic building projects.",
            "objective": f"Your objective is to generate a comprehensive technical report for the building \"{project_name}\" located in \"{project_location}\", using data from the IFC model loaded in Blender.",
            "workflow_title": "## MANDATORY WORKFLOW:",
            "report_language": "Write the entire report in English."
        },
        "spanish": {
            "role": "Eres un arquitecto técnico especializado en la creación de memorias técnicas de proyectos básicos de edificación.",
            "objective": f"Tu objetivo es generar una memoria técnica completa del edificio \"{project_name}\" localizado en \"{project_location}\", utilizando los datos del modelo IFC cargado en Blender.",
            "workflow_title": "## FLUJO DE TRABAJO OBLIGATORIO:",
            "report_language": "Redacta todo el informe en español."
        },
        "french": {
            "role": "Vous êtes un architecte technique spécialisé dans la création de rapports techniques pour les projets de bâtiment de base.",
            "objective": f"Votre objectif est de générer un rapport technique complet pour le bâtiment \"{project_name}\" situé à \"{project_location}\", en utilisant les données du modèle IFC chargé dans Blender.",
            "workflow_title": "## FLUX DE TRAVAIL OBLIGATOIRE:",
            "report_language": "Rédigez tout le rapport en français."
        },
        "german": {
            "role": "Sie sind ein technischer Architekt, der sich auf die Erstellung technischer Berichte für grundlegende Bauprojekte spezialisiert hat.",
            "objective": f"Ihr Ziel ist es, einen umfassenden technischen Bericht für das Gebäude \"{project_name}\" in \"{project_location}\" zu erstellen, unter Verwendung der Daten aus dem in Blender geladenen IFC-Modell.",
            "workflow_title": "## OBLIGATORISCHER ARBEITSABLAUF:",
            "report_language": "Verfassen Sie den gesamten Bericht auf Deutsch."
        },
        "italian": {
            "role": "Sei un architetto tecnico specializzato nella creazione di relazioni tecniche per progetti edilizi di base.",
            "objective": f"Il tuo obiettivo è generare una relazione tecnica completa per l'edificio \"{project_name}\" situato a \"{project_location}\", utilizzando i dati del modello IFC caricato in Blender.",
            "workflow_title": "## FLUSSO DI LAVORO OBBLIGATORIO:",
            "report_language": "Scrivi tutto il rapporto in italiano."
        },
        "portuguese": {
            "role": "Você é um arquiteto técnico especializado na criação de relatórios técnicos para projetos básicos de construção.",
            "objective": f"Seu objetivo é gerar um relatório técnico abrangente para o edifício \"{project_name}\" localizado em \"{project_location}\", usando dados do modelo IFC carregado no Blender.",
            "workflow_title": "## FLUXO DE TRABALHO OBRIGATÓRIO:",
            "report_language": "Escreva todo o relatório em português."
        }
    }
    
    # Get language instructions (default to English if language not supported)
    lang_config = language_instructions.get(language.lower(), language_instructions["english"])
    
    return f"""
{lang_config["role"]} {lang_config["objective"]}

**LANGUAGE REQUIREMENT:** {lang_config["report_language"]}

{lang_config["workflow_title"]}

### 1. INITIAL IFC MODEL ANALYSIS
- **Use MCP tool:** `get_ifc_project_info` to get basic project information
- **Use MCP tool:** `get_ifc_spatial_structure` to understand the building's spatial structure
- **Use MCP tool:** `get_user_view` to capture a general view of the model

### 2. OBTAIN TABLE OF CONTENTS
- **Access MCP resource:** `file://table_of_contents.md` to get the complete technical report structure

### 3. DETAILED ANALYSIS BY SECTIONS

#### 3.1 For "General Building Data" Section:
- **Use:** `get_ifc_quantities` to obtain areas and volumes
- **Use:** `list_ifc_entities` with entity_type="IfcSpace" for spaces
- **Use:** `list_ifc_entities` with entity_type="IfcBuildingStorey" for floors

#### 3.2 For "Architectural Solution" Section:
- **Use:** `list_ifc_entities` with entity_type="IfcWall" for walls
- **Use:** `list_ifc_entities` with entity_type="IfcDoor" for doors
- **Use:** `list_ifc_entities` with entity_type="IfcWindow" for windows
- **Use:** `get_user_view` to capture representative views

#### 3.3 For "Construction Systems" Section:
- **Use:** `list_ifc_entities` with entity_type="IfcBeam" for beams
- **Use:** `list_ifc_entities` with entity_type="IfcColumn" for columns
- **Use:** `list_ifc_entities` with entity_type="IfcSlab" for slabs
- **Use:** `list_ifc_entities` with entity_type="IfcRoof" for roofs
- **Use:** `get_ifc_properties` to obtain material properties

#### 3.4 For Building Services:
- **Use:** `list_ifc_entities` with entity_type="IfcPipeSegment" for plumbing
- **Use:** `list_ifc_entities` with entity_type="IfcCableSegment" for electrical
- **Use:** `list_ifc_entities` with entity_type="IfcDuctSegment" for HVAC

#### 3.5 For drawings and Graphic Documentation:
- **Use:** `export_drawing_png` 5 times, using as parameter each time "top", "front", "right", "left" and "isometric", to generate architectural drawings.
- **Configure:** resolution_x=1920, resolution_y=1080 for adequate quality
- **Use:** `get_user_view` for complementary 3D views

### 4. TECHNICAL REPORT STRUCTURE

Organize the document following exactly the structure from the `table_of_contents.md` resource:

**TECHNICAL REPORT – BASIC PROJECT: {project_name}**

**Location:** {project_location}

#### 1. INTRODUCTION
- Define object and scope based on IFC model data
- Justify the adopted architectural solution

#### 2. GENERAL BUILDING DATA
- **Location:** {project_location}
- **Areas:** Extract from quantities and spaces analysis
- **Distribution:** Based on IFC spatial structure
- **Regulations:** Identify applicable regulations according to use and location

#### 3-11. DEVELOPMENT OF ALL SECTIONS
- Complete each section according to the index, using data extracted from the IFC model
- Include summary tables of areas, materials and construction elements
- Generate technical conclusions based on evidence

### 5. MANDATORY GRAPHIC DOCUMENTATION
- **2D drawings:**Include the 4 2D drawings generated before in the 3.5 section with the Tool `export_drawing_png` ("top", "front", "right", "left")
- **3D views:** Include the isometric 3D view generated before in the 3.5 section with the Tool `export_drawing_png`
- **Organize:** All images in section 11. Annexes

### 6. TECHNICAL TABLES AND CHARTS
- **Areas summary table:** Extracted from quantities
- **Elements listing:** By typologies (walls, columns, beams, etc.)
- **Material properties:** From IFC properties

## RESPONSE FORMAT:

### MARKDOWN STRUCTURE:
```markdown
# TECHNICAL REPORT – BASIC PROJECT
## {project_name}

### Project Data:
- **Location:** {project_location}
- **Date:** [current date]
- **IFC Model:** [model information]

[Complete development of all index sections]
```

### QUALITY CRITERIA:
- **Technical precision:** All numerical data extracted directly from IFC model
- **Completeness:** Cover all index sections mandatory
- **Professional format:** Markdown tables, structured text, integrated images
- **Consistency:** Verify data consistency between sections

## CRITICAL VALIDATIONS:
1. **Verify Blender connection:** Confirm IFC model is loaded
2. **Complete all sections:** Do not omit any index section
3. **Include graphic documentation:** drawings and 3D views mandatory
4. **Quantitative data:** Areas, volumes and quantities verified
5. **Regulatory consistency:** Applicable regulations according to use and location

**IMPORTANT:** If any MCP tool fails or doesn't return data, document the limitation and indicate that section requires manual completion in executive project phase.

Proceed to generate the technical report following this detailed workflow.
"""


# Main execution

def main():
    """Run the MCP server"""
    mcp.run()

if __name__ == "__main__":
    main()