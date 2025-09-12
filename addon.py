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
            "export_ifc_data": self.export_ifc_data,
            "place_ifc_object": self.place_ifc_object,
            "get_ifc_quantities": self.get_ifc_quantities,
            "export_drawing_png": self.export_drawing_png,
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
        

    @staticmethod
    def export_ifc_data(entity_type=None, level_name=None, output_format="csv"):
        """Export IFC data to a structured file"""
        try:
            file = IfcStore.get_file()
            if file is None:
                return {"error": "No IFC file is currently loaded"}
            
            data_list = []
            
            # Filter objects based on type
            if entity_type:
                objects = file.by_type(entity_type)
            else:
                objects = file.by_type("IfcElement")
            
            # Create a data dictionary for each object
            for obj in objects:
                obj_data = {}
                
                # Get level/storey information
                container_level = None
                try:
                    containing_structure = ifcopenshell.util.element.get_container(obj)
                    if containing_structure and containing_structure.is_a("IfcBuildingStorey"):
                        container_level = containing_structure.Name
                except Exception as e:
                    pass
                
                # Skip if we're filtering by level and this doesn't match
                if level_name and container_level != level_name:
                    continue
                    
                # Basic information
                obj_data['ExpressId'] = obj.id()
                obj_data['GlobalId'] = obj.GlobalId if hasattr(obj, "GlobalId") else None
                obj_data['IfcClass'] = obj.is_a()
                obj_data['Name'] = obj.Name if hasattr(obj, "Name") else None
                obj_data['Description'] = obj.Description if hasattr(obj, "Description") else None
                obj_data['LevelName'] = container_level
                
                # Get predefined type if available
                try:
                    obj_data['PredefinedType'] = ifcopenshell.util.element.get_predefined_type(obj)
                except:
                    obj_data['PredefinedType'] = None
                    
                # Get type information
                try:
                    type_obj = ifcopenshell.util.element.get_type(obj)
                    obj_data['TypeName'] = type_obj.Name if type_obj and hasattr(type_obj, "Name") else None
                    obj_data['TypeClass'] = type_obj.is_a() if type_obj else None
                except:
                    obj_data['TypeName'] = None
                    obj_data['TypeClass'] = None
                
                # Get property sets (simplify structure for export)
                try:
                    property_sets = ifcopenshell.util.element.get_psets(obj)
                    # Flatten property sets for better export compatibility
                    for pset_name, pset_data in property_sets.items():
                        for prop_name, prop_value in pset_data.items():
                            obj_data[f"{pset_name}.{prop_name}"] = prop_value
                except Exception as e:
                    pass
                    
                data_list.append(obj_data)
            
            if not data_list:
                return "No data found matching the specified criteria"
            
            # Determine output directory - try multiple options to ensure it works in various environments
            output_dirs = [
                "C:\\Users\\Public\\Documents" if os.name == "nt" else None,  # Public Documents
                "/usr/share" if os.name != "nt" else None,  # Unix share directory
                "/tmp",  # Unix temp directory
                "C:\\Temp" if os.name == "nt" else None,  # Windows temp directory
            ]
            
            output_dir = None
            for dir_path in output_dirs:
                if dir_path and os.path.exists(dir_path) and os.access(dir_path, os.W_OK):
                    output_dir = dir_path
                    break
                    
            if not output_dir:
                return {"error": "Could not find a writable directory for output"}
            
            # Create filename based on filters
            filters = []
            if entity_type:
                filters.append(entity_type)
            if level_name:
                filters.append(level_name)
            filter_str = "_".join(filters) if filters else "all"
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"ifc_export_{filter_str}_{timestamp}.{output_format}"
            filepath = os.path.join(output_dir, filename)
            
            # Export based on format
            if output_format == "json":
                with open(filepath, 'w') as f:
                    json.dump(data_list, f, indent=2)
            elif output_format == "csv":
                import pandas as pd
                df = pd.DataFrame(data_list)
                df.to_csv(filepath, index=False)
            
            # Summary info for the response
            entity_count = len(data_list)
            entity_types = set(item['IfcClass'] for item in data_list)
            levels = set(item['LevelName'] for item in data_list if item['LevelName'])
            
            return {
                "success": True,
                "message": f"Data exported successfully to {filepath}",
                "filepath": filepath,
                "format": output_format,
                "summary": {
                    "entity_count": entity_count,
                    "entity_types": list(entity_types),
                    "levels": list(levels)
                }
            }
        
        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}
        
    
    @staticmethod
    def place_ifc_object(type_name, location, rotation=None):
        """
        Place an IFC object at specified location with optional rotation
        
        Args:
            type_name: Name of the IFC element type
            location: [x, y, z] list or tuple for position
            rotation: Value in degrees for rotation around Z axis (optional)
        
        Returns:
            Dictionary with information about the created object
        """
        try:
            import ifcopenshell
            from bonsai.bim.ifc import IfcStore
            import math
            
            # Convert location to tuple if it's not already
            if isinstance(location, list):
                location = tuple(location)
                
            def find_type_by_name(name):
                file = IfcStore.get_file()
                for element in file.by_type("IfcElementType"):
                    if element.Name == name:
                        return element.id()
                return None

            # Find the type ID
            type_id = find_type_by_name(type_name)
            if not type_id:
                return {"error": f"Type '{type_name}' not found. Please check if this type exists in the model."}
                
            # Store original context
            original_context = bpy.context.copy()
            
            # Ensure we're in 3D View context
            override = bpy.context.copy()
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    override["area"] = area
                    override["region"] = area.regions[-1]
                    break
            
            # Set cursor location
            bpy.context.scene.cursor.location = location
            
            # Get properties to set up parameters
            props = bpy.context.scene.BIMModelProperties
            
            # Store original rl_mode and set to CURSOR to use cursor's Z position
            original_rl_mode = props.rl_mode
            props.rl_mode = 'CURSOR'
            
            # Create the object using the override context
            with bpy.context.temp_override(**override):
                bpy.ops.bim.add_occurrence(relating_type_id=type_id)
            
            # Get the newly created object
            obj = bpy.context.active_object
            if not obj:
                props.rl_mode = original_rl_mode
                return {"error": "Failed to create object"}
            
            # Force the Z position explicitly
            obj.location.z = location[2]
            
            # Apply rotation if provided
            if rotation is not None:
                # Convert degrees to radians for Blender's rotation_euler
                full_rotation = (0, 0, math.radians(float(rotation)))
                obj.rotation_euler = full_rotation
            
            # Sync the changes back to IFC
            # Use the appropriate method depending on what's available
            if hasattr(bpy.ops.bim, "update_representation"):
                bpy.ops.bim.update_representation(obj=obj.name)
            
            # Restore original rl_mode
            props.rl_mode = original_rl_mode
            
            # Get the IFC entity for the new object
            entity_id = obj.BIMObjectProperties.ifc_definition_id
            if entity_id:
                file = IfcStore.get_file()
                entity = file.by_id(entity_id)
                global_id = entity.GlobalId if hasattr(entity, "GlobalId") else None
            else:
                global_id = None
            
            # Return information about the created object
            return {
                "success": True,
                "blender_name": obj.name,
                "global_id": global_id,
                "location": list(obj.location),
                "rotation": list(obj.rotation_euler),
                "type_name": type_name
            }
            
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


    @staticmethod
    def get_ifc_quantities(entity_type=None, selected_only=False):
        """
        Calculate and get quantities (m2, m3, etc.) for IFC elements.
        
        Parameters:
            entity_type: Type of IFC entity to get quantities for (e.g., "IfcWall", "IfcSlab")
            selected_only: If True, only get quantities for selected objects
        
        Returns:
            Dictionary with quantities for the specified elements
        """
        try:
            file = IfcStore.get_file()
            if file is None:
                return {"error": "No IFC file is currently loaded"}
            
            # First, calculate all quantities
            try:
                bpy.ops.bim.perform_quantity_take_off()
            except Exception as e:
                return {"error": f"Failed to calculate quantities: {str(e)}"}
            
            elements_data = []
            
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
                for entity_info in selected_result["selected_entities"]:
                    # Find entity by GlobalId
                    entity = file.by_guid(entity_info["id"])
                    if not entity:
                        continue
                    
                    # Filter by type if specified
                    if entity_type and entity.is_a() != entity_type:
                        continue
                    
                    # Extract quantities
                    element_data = extract_quantities(entity, entity_info["blender_name"])
                    if element_data:
                        elements_data.append(element_data)
                        
            else:
                # Get entities based on type or default to common element types
                if entity_type:
                    entities = file.by_type(entity_type)
                else:
                    # Get common element types that have quantities
                    entity_types = ["IfcWall", "IfcSlab", "IfcBeam", "IfcColumn", "IfcDoor", "IfcWindow"]
                    entities = []
                    for etype in entity_types:
                        entities.extend(file.by_type(etype))
                
                # Process each entity
                for entity in entities:
                    element_data = extract_quantities(entity)
                    if element_data:
                        elements_data.append(element_data)
            
            # Summary statistics
            summary = {
                "total_elements": len(elements_data),
                "element_types": {}
            }
            
            # Group by element type for summary
            for element in elements_data:
                etype = element["type"]
                if etype not in summary["element_types"]:
                    summary["element_types"][etype] = {"count": 0, "total_area": 0, "total_volume": 0}
                
                summary["element_types"][etype]["count"] += 1
                if element["quantities"].get("area"):
                    summary["element_types"][etype]["total_area"] += element["quantities"]["area"]
                if element["quantities"].get("volume"):
                    summary["element_types"][etype]["total_volume"] += element["quantities"]["volume"]
            
            return {
                "success": True,
                "elements": elements_data,
                "summary": summary
            }
            
        except Exception as e:
            import traceback
            return {"error": str(e), "traceback": traceback.format_exc()}
    
    @staticmethod
    def export_drawing_png(view_type="top", height_offset=0.5, resolution_x=1920, 
                             resolution_y=1080, storey_name=None, output_path=None):
        """
        Export drawings as PNG images with custom resolution.
        
        Creates 2D and 3D views of IFC building, particularly useful for architectural drawings.
        
        Args:
            view_type: "top" for plan view, "front", "right", "left" for elevations, "isometric" for 3D view
            height_offset: Height in meters above storey level for camera position  
            resolution_x: Horizontal resolution in pixels
            resolution_y: Vertical resolution in pixels
            storey_name: Specific storey name to render (None for all/ground floor)
            output_path: File path to save PNG (None for temp file)
        
        Returns:
            Dict with base64 encoded image data and metadata
        """
        try:
            import tempfile
            import os
            
            # Validate parameters
            if resolution_x > 4096 or resolution_y > 4096:
                return {"error": "Resolution too high. Maximum: 4096x4096"}
            
            if resolution_x < 100 or resolution_y < 100:
                return {"error": "Resolution too low. Minimum: 100x100"}
            
            # Check if IFC file is loaded
            file = IfcStore.get_file()
            if file is None:
                return {"error": "No IFC file is currently loaded"}
            
            # Store original render settings
            scene = bpy.context.scene
            original_engine = scene.render.engine
            original_res_x = scene.render.resolution_x
            original_res_y = scene.render.resolution_y
            original_filepath = scene.render.filepath
            
            # Set up render settings for drawing
            scene.render.engine = 'BLENDER_WORKBENCH'  # Fast, good for architectural drawings
            scene.render.resolution_x = resolution_x
            scene.render.resolution_y = resolution_y
            scene.render.resolution_percentage = 100
            
            # Store original camera if exists
            original_camera = bpy.context.scene.camera
            
            # Create temporary camera for orthographic rendering
            bpy.ops.object.camera_add()
            camera = bpy.context.object
            camera.name = "TempDrawingCamera"
            bpy.context.scene.camera = camera
            
            # Set camera to orthographic
            camera.data.type = 'ORTHO'
            camera.data.ortho_scale = 50  # Adjust based on building size
            
            # Position camera based on view type and storey
            if view_type == "top":
                # Find building bounds to position camera appropriately
                all_objects = [obj for obj in bpy.context.scene.objects 
                              if obj.type == 'MESH' and obj.visible_get()]
                
                if all_objects:
                    # Calculate bounding box of all visible objects
                    min_x = min_y = min_z = float('inf')
                    max_x = max_y = max_z = float('-inf')
                    
                    for obj in all_objects:
                        bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
                        for corner in bbox:
                            min_x = min(min_x, corner.x)
                            max_x = max(max_x, corner.x)
                            min_y = min(min_y, corner.y)  
                            max_y = max(max_y, corner.y)
                            min_z = min(min_z, corner.z)
                            max_z = max(max_z, corner.z)
                    
                    # Position camera above the building
                    center_x = (min_x + max_x) / 2
                    center_y = (min_y + max_y) / 2
                    
                    # For plan view, position camera above
                    camera_height = max_z + height_offset
                    camera.location = (center_x, center_y, camera_height)
                    camera.rotation_euler = (0, 0, 0)  # Look down
                    
                    # Adjust orthographic scale based on building size
                    building_width = max(max_x - min_x, max_y - min_y) * 1.2  # Add 20% margin
                    camera.data.ortho_scale = building_width
                else:
                    # Default position if no objects found
                    camera.location = (0, 0, 10)
                    camera.rotation_euler = (0, 0, 0)
            
            elif view_type in ["front", "right", "left"]:
                # For elevations, position camera accordingly
                # This is a simplified implementation - could be enhanced
                all_objects = [obj for obj in bpy.context.scene.objects 
                              if obj.type == 'MESH' and obj.visible_get()]
                
                if all_objects:
                    # Calculate bounds
                    min_x = min_y = min_z = float('inf')
                    max_x = max_y = max_z = float('-inf')
                    
                    for obj in all_objects:
                        bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
                        for corner in bbox:
                            min_x = min(min_x, corner.x)
                            max_x = max(max_x, corner.x)
                            min_y = min(min_y, corner.y)
                            max_y = max(max_y, corner.y)
                            min_z = min(min_z, corner.z)
                            max_z = max(max_z, corner.z)
                    
                    center_x = (min_x + max_x) / 2
                    center_y = (min_y + max_y) / 2
                    center_z = (min_z + max_z) / 2
                    
                    building_depth = max(max_x - min_x, max_y - min_y) * 2
                    
                    if view_type == "front":
                        camera.location = (center_x, center_y - building_depth, center_z)
                        camera.rotation_euler = (1.5708, 0, 0)  # 90 degrees X rotation
                    elif view_type == "right":
                        camera.location = (center_x + building_depth, center_y, center_z)
                        camera.rotation_euler = (1.5708, 0, 1.5708)  # Look from right
                    elif view_type == "left":
                        camera.location = (center_x - building_depth, center_y, center_z)
                        camera.rotation_euler = (1.5708, 0, -1.5708)  # Look from left
                    
                    # Adjust scale for elevations
                    building_height = max_z - min_z
                    building_width = max(max_x - min_x, max_y - min_y)
                    camera.data.ortho_scale = max(building_height, building_width) * 1.2
            
            elif view_type == "isometric":
                # For isometric view, use perspective camera positioned diagonally
                camera.data.type = 'PERSP'
                camera.data.lens = 35  # 35mm lens for nice perspective
                
                all_objects = [obj for obj in bpy.context.scene.objects 
                              if obj.type == 'MESH' and obj.visible_get()]
                
                if all_objects:
                    # Calculate bounds
                    min_x = min_y = min_z = float('inf')
                    max_x = max_y = max_z = float('-inf')
                    
                    for obj in all_objects:
                        bbox = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
                        for corner in bbox:
                            min_x = min(min_x, corner.x)
                            max_x = max(max_x, corner.x)
                            min_y = min(min_y, corner.y)
                            max_y = max(max_y, corner.y)
                            min_z = min(min_z, corner.z)
                            max_z = max(max_z, corner.z)
                    
                    center_x = (min_x + max_x) / 2
                    center_y = (min_y + max_y) / 2
                    center_z = (min_z + max_z) / 2
                    
                    # Calculate distance to frame the building nicely
                    building_size = max(max_x - min_x, max_y - min_y, max_z - min_z)
                    distance = building_size * 1.2  # Distance multiplier for good framing
                    
                    # Position camera for isometric view (45° angles)
                    # Classic isometric position: up and back, looking down at 30°
                    import math
                    angle_rad = math.radians(45)
                    
                    camera_x = center_x + distance * math.cos(angle_rad)
                    camera_y = center_y - distance * math.sin(angle_rad)
                    camera_z = center_z + distance * 0.3  # Lower elevation for better facade view
                    
                    camera.location = (camera_x, camera_y, camera_z)
                    
                    # Point camera at building center
                    direction = mathutils.Vector((center_x - camera_x, center_y - camera_y, center_z - camera_z))
                    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
                else:
                    # Default isometric position
                    camera.location = (15, -15, 10)
                    camera.rotation_euler = (1.1, 0, 0.785)  # ~63°, 0°, ~45°
            
            # Set up output file path
            if output_path:
                render_path = output_path
            else:
                temp_dir = tempfile.gettempdir()
                render_path = os.path.join(temp_dir, f"drawing_{view_type}_{int(time.time())}.png")
            
            scene.render.filepath = render_path
            scene.render.image_settings.file_format = 'PNG'
            
            # Render the image
            bpy.ops.render.render(write_still=True)
            
            # Read the rendered image and encode as base64
            if os.path.exists(render_path):
                with open(render_path, 'rb') as f:
                    image_data = f.read()
                
                # Clean up temporary file if we created it
                if not output_path:
                    os.remove(render_path)
                
                # Restore original settings
                scene.render.engine = original_engine
                scene.render.resolution_x = original_res_x
                scene.render.resolution_y = original_res_y
                scene.render.filepath = original_filepath
                bpy.context.scene.camera = original_camera
                
                # Delete temporary camera
                bpy.data.objects.remove(camera, do_unlink=True)
                
                # Return base64 encoded image
                import base64
                return {
                    "success": True,
                    "data": base64.b64encode(image_data).decode('utf-8'),
                    "format": "png",
                    "resolution": f"{resolution_x}x{resolution_y}",
                    "view_type": view_type,
                    "output_path": render_path if output_path else None
                }
            else:
                return {"error": "Failed to create render file"}
                
        except Exception as e:
            # Restore settings on error
            try:
                scene = bpy.context.scene
                scene.render.engine = original_engine
                scene.render.resolution_x = original_res_x
                scene.render.resolution_y = original_res_y 
                scene.render.filepath = original_filepath
                bpy.context.scene.camera = original_camera
                
                # Clean up camera if it exists
                if 'camera' in locals() and camera:
                    bpy.data.objects.remove(camera, do_unlink=True)
            except:
                pass
                
            import traceback
            return {"error": f"Error creating drawing: {str(e)}", 
                    "traceback": traceback.format_exc()}


    #endregion


def extract_quantities(entity, blender_name=None):
    """
    Extract quantity information from an IFC entity.
    
    Parameters:
        entity: IFC entity object
        blender_name: Optional Blender object name
    
    Returns:
        Dictionary with element info and quantities
    """
    try:
        # Get all property sets
        psets = ifcopenshell.util.element.get_psets(entity)
        
        # Basic element info
        element_data = {
            "id": entity.GlobalId if hasattr(entity, "GlobalId") else f"Entity_{entity.id()}",
            "name": entity.Name if hasattr(entity, "Name") else None,
            "type": entity.is_a(),
            "blender_name": blender_name,
            "quantities": {}
        }
        
        # Look for quantity information in different property sets
        quantity_sources = ["BaseQuantities", "ArchiCADQuantities", "Qto_WallBaseQuantities", 
                           "Qto_SlabBaseQuantities", "Qto_BeamBaseQuantities", "Qto_ColumnBaseQuantities"]
        
        # Common quantity mappings
        quantity_mappings = {
            # Area measurements
            "GrossSideArea": "area",
            "NetSideArea": "net_area", 
            "GrossFootprintArea": "footprint_area",
            "NetFootprintArea": "net_footprint_area",
            "Oberflächenbereich": "surface_area",
            
            # Volume measurements
            "GrossVolume": "volume",
            "NetVolume": "net_volume",
            "Netto-Volumen": "net_volume_de",
            "Brutto-Volumen der Wand ": "gross_volume_de",
            
            # Length measurements
            "Length": "length",
            "Height": "height",
            "Width": "width",
            "Höhe": "height_de",
            "Dicke": "thickness",
            
            # Other measurements
            "Fläche": "area_de",
            "Wandlänge an der Außenseite": "outer_length",
            "Wandlänge an der Innenseite": "inner_length"
        }
        
        # Extract quantities from property sets
        for pset_name in quantity_sources:
            if pset_name in psets:
                pset_data = psets[pset_name]
                for prop_name, prop_value in pset_data.items():
                    if prop_name in quantity_mappings and isinstance(prop_value, (int, float)):
                        mapped_name = quantity_mappings[prop_name]
                        element_data["quantities"][mapped_name] = prop_value
        
        # Add common derived quantities if available
        if "area" in element_data["quantities"]:
            element_data["quantities"]["area_m2"] = element_data["quantities"]["area"]
        if "volume" in element_data["quantities"]:
            element_data["quantities"]["volume_m3"] = element_data["quantities"]["volume"]
            
        return element_data if element_data["quantities"] else None
        
    except Exception as e:
        return None


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

    print("BlenderMCP addon unregistered")

if __name__ == "__main__":
    register()
