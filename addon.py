import bpy
import mathutils
import json
import threading
import socket
import time
import requests
import tempfile
import traceback
import os
import shutil
from bpy.props import StringProperty, IntProperty, BoolProperty, EnumProperty
import base64

import bpy

import ifcopenshell
from bonsai.bim.ifc import IfcStore

bl_info = {
    "name": "Bonsai MCP",
    "author": "JotaDeRodriguez",
    "version": (0, 2),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Bonsai MCP",
    "description": "Connect Claude to Blender via MCP. Aimed at IFC projects",
    "category": "Interface",
}


class BlenderMCPServer:
    def __init__(self, host='localhost', port=9876):
        self.host = host
        self.port = port
        self.running = False
        self.socket = None
        self.server_thread = None
    
    def start(self):
        if self.running:
            print("Server is already running")
            return
            
        self.running = True
        
        try:
            # Create socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((self.host, self.port))
            self.socket.listen(1)
            
            # Start server thread
            self.server_thread = threading.Thread(target=self._server_loop)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            print(f"BlenderMCP server started on {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to start server: {str(e)}")
            self.stop()
            
    def stop(self):
        self.running = False
        
        # Close socket
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        
        # Wait for thread to finish
        if self.server_thread:
            try:
                if self.server_thread.is_alive():
                    self.server_thread.join(timeout=1.0)
            except:
                pass
            self.server_thread = None
        
        print("BlenderMCP server stopped")
    
    def _server_loop(self):
        """Main server loop in a separate thread"""
        print("Server thread started")
        self.socket.settimeout(1.0)  # Timeout to allow for stopping
        
        while self.running:
            try:
                # Accept new connection
                try:
                    client, address = self.socket.accept()
                    print(f"Connected to client: {address}")
                    
                    # Handle client in a separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client,)
                    )
                    client_thread.daemon = True
                    client_thread.start()
                except socket.timeout:
                    # Just check running condition
                    continue
                except Exception as e:
                    print(f"Error accepting connection: {str(e)}")
                    time.sleep(0.5)
            except Exception as e:
                print(f"Error in server loop: {str(e)}")
                if not self.running:
                    break
                time.sleep(0.5)
        
        print("Server thread stopped")
    
    def _handle_client(self, client):
        """Handle connected client"""
        print("Client handler started")
        client.settimeout(None)  # No timeout
        buffer = b''
        
        try:
            while self.running:
                # Receive data
                try:
                    data = client.recv(8192)
                    if not data:
                        print("Client disconnected")
                        break
                    
                    buffer += data
                    try:
                        # Try to parse command
                        command = json.loads(buffer.decode('utf-8'))
                        buffer = b''
                        
                        # Execute command in Blender's main thread
                        def execute_wrapper():
                            try:
                                response = self.execute_command(command)
                                response_json = json.dumps(response)
                                try:
                                    client.sendall(response_json.encode('utf-8'))
                                except:
                                    print("Failed to send response - client disconnected")
                            except Exception as e:
                                print(f"Error executing command: {str(e)}")
                                traceback.print_exc()
                                try:
                                    error_response = {
                                        "status": "error",
                                        "message": str(e)
                                    }
                                    client.sendall(json.dumps(error_response).encode('utf-8'))
                                except:
                                    pass
                            return None
                        
                        # Schedule execution in main thread
                        bpy.app.timers.register(execute_wrapper, first_interval=0.0)
                    except json.JSONDecodeError:
                        # Incomplete data, wait for more
                        pass
                except Exception as e:
                    print(f"Error receiving data: {str(e)}")
                    break
        except Exception as e:
            print(f"Error in client handler: {str(e)}")
        finally:
            try:
                client.close()
            except:
                pass
            print("Client handler stopped")

    def execute_command(self, command):
        """Execute a command in the main Blender thread"""
        try:
            cmd_type = command.get("type")
            params = command.get("params", {})
            
            # Ensure we're in the right context
            if cmd_type in ["create_object", "modify_object", "delete_object"]:
                override = bpy.context.copy()
                override['area'] = [area for area in bpy.context.screen.areas if area.type == 'VIEW_3D'][0]
                with bpy.context.temp_override(**override):
                    return self._execute_command_internal(command)
            else:
                return self._execute_command_internal(command)
                
        except Exception as e:
            print(f"Error executing command: {str(e)}")
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    def _execute_command_internal(self, command):
        """Internal command execution with proper context"""
        cmd_type = command.get("type")
        params = command.get("params", {})

        
        # Base handlers that are always available
        handlers = {
            "execute_code": self.execute_code,
            "get_ifc_project_info": self.get_ifc_project_info,
            "list_ifc_entities": self.list_ifc_entities,
            "get_ifc_properties": self.get_ifc_properties,
            "get_ifc_spatial_structure": self.get_ifc_spatial_structure,
            "get_ifc_relationships": self.get_ifc_relationships,
            "get_selected_ifc_entities": self.get_selected_ifc_entities,
            "get_current_view": self.get_current_view,
            "create_orthographic_render": self.create_orthographic_render
        }
        

        handler = handlers.get(cmd_type)
        if handler:
            try:
                print(f"Executing handler for {cmd_type}")
                result = handler(**params)
                print(f"Handler execution complete")
                return {"status": "success", "result": result}
            except Exception as e:
                print(f"Error in handler: {str(e)}")
                traceback.print_exc()
                return {"status": "error", "message": str(e)}
        else:
            return {"status": "error", "message": f"Unknown command type: {cmd_type}"}

    
    def execute_code(self, code):
        """Execute arbitrary Blender Python code"""
        # This is powerful but potentially dangerous - use with caution
        try:
            # Create a local namespace for execution
            namespace = {"bpy": bpy}
            exec(code, namespace)
            return {"executed": True}
        except Exception as e:
            raise Exception(f"Code execution error: {str(e)}")
        

    @staticmethod
    def get_selected_ifc_entities():
        """
        Get the IFC entities corresponding to the currently selected Blender objects.
        
        Returns:
            List of IFC entities for the selected objects
        """
        try:
            file = IfcStore.get_file()
            if file is None:
                return {"error": "No IFC file is currently loaded"}
            
            # Get currently selected objects
            selected_objects = bpy.context.selected_objects
            if not selected_objects:
                return {"selected_count": 0, "message": "No objects selected in Blender"}
            
            # Collect IFC entities from selected objects
            selected_entities = []
            for obj in selected_objects:
                if hasattr(obj, "BIMObjectProperties") and obj.BIMObjectProperties.ifc_definition_id:
                    entity_id = obj.BIMObjectProperties.ifc_definition_id
                    entity = file.by_id(entity_id)
                    if entity:
                        entity_info = {
                            "id": entity.GlobalId if hasattr(entity, "GlobalId") else f"Entity_{entity.id()}",
                            "ifc_id": entity.id(),
                            "type": entity.is_a(),
                            "name": entity.Name if hasattr(entity, "Name") else None,
                            "blender_name": obj.name
                        }
                        selected_entities.append(entity_info)
            
            return {
                "selected_count": len(selected_entities),
                "selected_entities": selected_entities
            }
        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}
        
    ### SPECIFIC IFC METHODS ###
        
    @staticmethod
    def get_ifc_project_info():
        """
        Get basic information about the IFC project.
        
        Returns:
            Dictionary with project name, description, and basic metrics
        """
        try:
            file = IfcStore.get_file()
            if file is None:
                return {"error": "No IFC file is currently loaded"}
            
            # Get project information
            projects = file.by_type("IfcProject")
            if not projects:
                return {"error": "No IfcProject found in the model"}
            
            project = projects[0]
            
            # Basic project info
            info = {
                "id": project.GlobalId,
                "name": project.Name if hasattr(project, "Name") else "Unnamed Project",
                "description": project.Description if hasattr(project, "Description") else None,
                "entity_counts": {}
            }
            
            # Count entities by type
            entity_types = ["IfcWall", "IfcDoor", "IfcWindow", "IfcSlab", "IfcBeam", "IfcColumn", "IfcSpace", "IfcBuildingStorey"]
            for entity_type in entity_types:
                entities = file.by_type(entity_type)
                info["entity_counts"][entity_type] = len(entities)
            
            return info
        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}

    @staticmethod
    def list_ifc_entities(entity_type=None, limit=50, selected_only=False):
        """
        List IFC entities of a specific type.
        
        Parameters:
            entity_type: Type of IFC entity to list (e.g., "IfcWall")
            limit: Maximum number of entities to return
        
        Returns:
            List of entities with basic properties
        """
        try:
            file = IfcStore.get_file()
            if file is None:
                return {"error": "No IFC file is currently loaded"}
            
            # If we're only looking at selected objects
            if selected_only:
                selected_result = BlenderMCPServer.get_selected_ifc_entities()
                
                # Check for errors
                if "error" in selected_result:
                    return selected_result
                    
                # If no objects are selected, return early
                if selected_result["selected_count"] == 0:
                    return selected_result
                    
                # If entity_type is specified, filter the selected entities
                if entity_type:
                    filtered_entities = [
                        entity for entity in selected_result["selected_entities"]
                        if entity["type"] == entity_type
                    ]
                    
                    return {
                        "type": entity_type,
                        "selected_count": len(filtered_entities),
                        "entities": filtered_entities[:limit]
                    }
                else:
                    # Group selected entities by type
                    entity_types = {}
                    for entity in selected_result["selected_entities"]:
                        entity_type = entity["type"]
                        if entity_type in entity_types:
                            entity_types[entity_type].append(entity)
                        else:
                            entity_types[entity_type] = [entity]
                    
                    return {
                        "selected_count": selected_result["selected_count"],
                        "entity_types": [
                            {"type": t, "count": len(entities), "entities": entities[:limit]}
                            for t, entities in entity_types.items()
                        ]
                    }
            
            # Original functionality for non-selected mode
            if not entity_type:
                # If no type specified, list available entity types
                entity_types = {}
                for entity in file.wrapped_data.entities:
                    entity_type = entity.is_a()
                    if entity_type in entity_types:
                        entity_types[entity_type] += 1
                    else:
                        entity_types[entity_type] = 1
                
                return {
                    "available_types": [{"type": k, "count": v} for k, v in entity_types.items()]
                }
            
            # Get entities of the specified type
            entities = file.by_type(entity_type)
            
            # Prepare the result
            result = {
                "type": entity_type,
                "total_count": len(entities),
                "entities": []
            }
            
            # Add entity data (limited)
            for i, entity in enumerate(entities):
                if i >= limit:
                    break
                    
                entity_data = {
                    "id": entity.GlobalId if hasattr(entity, "GlobalId") else f"Entity_{entity.id()}",
                    "name": entity.Name if hasattr(entity, "Name") else None
                }
                
                result["entities"].append(entity_data)
            
            return result
        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}

    @staticmethod
    def get_ifc_properties(global_id=None, selected_only=False):
        """
        Get all properties of a specific IFC entity.
        
        Parameters:
            global_id: GlobalId of the IFC entity
        
        Returns:
            Dictionary with entity information and properties
        """
        try:
            file = IfcStore.get_file()
            if file is None:
                return {"error": "No IFC file is currently loaded"}
            
            # If we're only looking at selected objects
            if selected_only:
                selected_result = BlenderMCPServer.get_selected_ifc_entities()
                
                # Check for errors
                if "error" in selected_result:
                    return selected_result
                
                # If no objects are selected, return early
                if selected_result["selected_count"] == 0:
                    return selected_result
                
                # Process each selected entity
                result = {
                    "selected_count": selected_result["selected_count"],
                    "entities": []
                }
                
                for entity_info in selected_result["selected_entities"]:
                    # Find entity by GlobalId
                    entity = file.by_guid(entity_info["id"])
                    if not entity:
                        continue
                    
                    # Get basic entity info
                    entity_data = {
                        "id": entity.GlobalId,
                        "type": entity.is_a(),
                        "name": entity.Name if hasattr(entity, "Name") else None,
                        "description": entity.Description if hasattr(entity, "Description") else None,
                        "blender_name": entity_info["blender_name"],
                        "property_sets": {}
                    }
                    
                    # Get all property sets
                    psets = ifcopenshell.util.element.get_psets(entity)
                    for pset_name, pset_data in psets.items():
                        entity_data["property_sets"][pset_name] = pset_data
                    
                    result["entities"].append(entity_data)
                
                return result
                
            # If we're looking at a specific entity
            elif global_id:
                # Find entity by GlobalId
                entity = file.by_guid(global_id)
                if not entity:
                    return {"error": f"No entity found with GlobalId: {global_id}"}
                
                # Get basic entity info
                entity_info = {
                    "id": entity.GlobalId,
                    "type": entity.is_a(),
                    "name": entity.Name if hasattr(entity, "Name") else None,
                    "description": entity.Description if hasattr(entity, "Description") else None,
                    "property_sets": {}
                }
                
                # Get all property sets
                psets = ifcopenshell.util.element.get_psets(entity)
                for pset_name, pset_data in psets.items():
                    entity_info["property_sets"][pset_name] = pset_data
                
                return entity_info
            else:
                return {"error": "Either global_id or selected_only must be specified"}
        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}

    @staticmethod
    def get_ifc_spatial_structure():
        """
        Get the spatial structure of the IFC model (site, building, storey, space hierarchy).
        
        Returns:
            Hierarchical structure of the IFC model's spatial elements
        """
        try:
            file = IfcStore.get_file()
            if file is None:
                return {"error": "No IFC file is currently loaded"}
            
            # Start with projects
            projects = file.by_type("IfcProject")
            if not projects:
                return {"error": "No IfcProject found in the model"}
            
            def get_children(parent):
                """Get immediate children of the given element"""
                if hasattr(parent, "IsDecomposedBy"):
                    rel_aggregates = parent.IsDecomposedBy
                    children = []
                    for rel in rel_aggregates:
                        children.extend(rel.RelatedObjects)
                    return children
                return []
                
            def create_structure(element):
                """Recursively create the structure for an element"""
                result = {
                    "id": element.GlobalId,
                    "type": element.is_a(),
                    "name": element.Name if hasattr(element, "Name") else None,
                    "children": []
                }
                
                for child in get_children(element):
                    result["children"].append(create_structure(child))
                
                return result
            
            # Create the structure starting from the project
            structure = create_structure(projects[0])
            
            return structure
        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}

    @staticmethod
    def get_ifc_relationships(global_id):
        """
        Get all relationships for a specific IFC entity.
        
        Parameters:
            global_id: GlobalId of the IFC entity
        
        Returns:
            Dictionary with all relationships the entity participates in
        """
        try:
            file = IfcStore.get_file()
            if file is None:
                return {"error": "No IFC file is currently loaded"}
            
            # Find entity by GlobalId
            entity = file.by_guid(global_id)
            if not entity:
                return {"error": f"No entity found with GlobalId: {global_id}"}
            
            # Basic entity info
            entity_info = {
                "id": entity.GlobalId,
                "type": entity.is_a(),
                "name": entity.Name if hasattr(entity, "Name") else None,
                "relationships": {
                    "contains": [],
                    "contained_in": [],
                    "connects": [],
                    "connected_by": [],
                    "defines": [],
                    "defined_by": []
                }
            }
            
            # Check if entity contains other elements
            if hasattr(entity, "IsDecomposedBy"):
                for rel in entity.IsDecomposedBy:
                    for obj in rel.RelatedObjects:
                        entity_info["relationships"]["contains"].append({
                            "id": obj.GlobalId,
                            "type": obj.is_a(),
                            "name": obj.Name if hasattr(obj, "Name") else None
                        })
            
            # Check if entity is contained in other elements
            if hasattr(entity, "Decomposes"):
                for rel in entity.Decomposes:
                    rel_obj = rel.RelatingObject
                    entity_info["relationships"]["contained_in"].append({
                        "id": rel_obj.GlobalId,
                        "type": rel_obj.is_a(),
                        "name": rel_obj.Name if hasattr(rel_obj, "Name") else None
                    })
            
            # For physical connections (depends on entity type)
            if hasattr(entity, "ConnectedTo"):
                for rel in entity.ConnectedTo:
                    for obj in rel.RelatedElement:
                        entity_info["relationships"]["connects"].append({
                            "id": obj.GlobalId,
                            "type": obj.is_a(),
                            "name": obj.Name if hasattr(obj, "Name") else None,
                            "connection_type": rel.ConnectionType if hasattr(rel, "ConnectionType") else None
                        })
            
            if hasattr(entity, "ConnectedFrom"):
                for rel in entity.ConnectedFrom:
                    obj = rel.RelatingElement
                    entity_info["relationships"]["connected_by"].append({
                        "id": obj.GlobalId,
                        "type": obj.is_a(),
                        "name": obj.Name if hasattr(obj, "Name") else None,
                        "connection_type": rel.ConnectionType if hasattr(rel, "ConnectionType") else None
                    })
            
            return entity_info
        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}
        
    
    ### Ability to see
    @staticmethod
    def get_current_view():
        """Capture and return the current viewport as an image"""
        try:
            # Find a 3D View
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    break
            else:
                return {"error": "No 3D View available"}
            
            # Create temporary file to save the viewport screenshot
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # Find appropriate region
            for region in area.regions:
                if region.type == 'WINDOW':
                    break
            else:
                return {"error": "No appropriate region found in 3D View"}
            
            # Use temp_override instead of the old override dictionary
            with bpy.context.temp_override(area=area, region=region):
                # Save screenshot
                bpy.ops.screen.screenshot(filepath=temp_path)
            
            # Read the image data and encode as base64
            with open(temp_path, 'rb') as f:
                image_data = f.read()
            
            # Clean up
            os.unlink(temp_path)
            
            # Return base64 encoded image
            return {
                "width": area.width,
                "height": area.height,
                "format": "png",
                "data": base64.b64encode(image_data).decode('utf-8')
            }
        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}



    #endregion

# Blender UI Panel
class BLENDERMCP_PT_Panel(bpy.types.Panel):
    bl_label = "Bonsai MCP"
    bl_idname = "BLENDERMCP_PT_Panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Bonsai MCP'
    
    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        layout.prop(scene, "blendermcp_port")
        
        if not scene.blendermcp_server_running:
            layout.operator("blendermcp.start_server", text="Start MCP Server")
        else:
            layout.operator("blendermcp.stop_server", text="Stop MCP Server")
            layout.label(text=f"Running on port {scene.blendermcp_port}")


# Operator to start the server
class BLENDERMCP_OT_StartServer(bpy.types.Operator):
    bl_idname = "blendermcp.start_server"
    bl_label = "Connect to Claude"
    bl_description = "Start the BlenderMCP server to connect with Claude"
    
    def execute(self, context):
        scene = context.scene
        
        # Create a new server instance
        if not hasattr(bpy.types, "blendermcp_server") or not bpy.types.blendermcp_server:
            bpy.types.blendermcp_server = BlenderMCPServer(port=scene.blendermcp_port)
        
        # Start the server
        bpy.types.blendermcp_server.start()
        scene.blendermcp_server_running = True
        
        return {'FINISHED'}

# Operator to stop the server
class BLENDERMCP_OT_StopServer(bpy.types.Operator):
    bl_idname = "blendermcp.stop_server"
    bl_label = "Stop the connection to Claude"
    bl_description = "Stop the connection to Claude"
    
    def execute(self, context):
        scene = context.scene
        
        # Stop the server if it exists
        if hasattr(bpy.types, "blendermcp_server") and bpy.types.blendermcp_server:
            bpy.types.blendermcp_server.stop()
            del bpy.types.blendermcp_server
        
        scene.blendermcp_server_running = False
        
        return {'FINISHED'}

# Registration functions
def register():
    bpy.types.Scene.blendermcp_port = IntProperty(
        name="Port",
        description="Port for the BlenderMCP server",
        default=9876,
        min=1024,
        max=65535
    )
    
    bpy.types.Scene.blendermcp_server_running = bpy.props.BoolProperty(
        name="Server Running",
        default=False
    )
    
    
    bpy.utils.register_class(BLENDERMCP_PT_Panel)
    bpy.utils.register_class(BLENDERMCP_OT_StartServer)
    bpy.utils.register_class(BLENDERMCP_OT_StopServer)
    
    print("BlenderMCP addon registered")

def unregister():
    # Stop the server if it's running
    if hasattr(bpy.types, "blendermcp_server") and bpy.types.blendermcp_server:
        bpy.types.blendermcp_server.stop()
        del bpy.types.blendermcp_server
    
    bpy.utils.unregister_class(BLENDERMCP_PT_Panel)
    bpy.utils.unregister_class(BLENDERMCP_OT_StartServer)
    bpy.utils.unregister_class(BLENDERMCP_OT_StopServer)
    
    del bpy.types.Scene.blendermcp_port
    del bpy.types.Scene.blendermcp_server_running
    del bpy.types.Scene.blendermcp_use_polyhaven
    del bpy.types.Scene.blendermcp_use_hyper3d
    del bpy.types.Scene.blendermcp_hyper3d_mode
    del bpy.types.Scene.blendermcp_hyper3d_api_key

    print("BlenderMCP addon unregistered")

if __name__ == "__main__":
    register()
