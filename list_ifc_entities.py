from simple_server import get_blender_connection
import json

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
        return f"Error listing IFC entities: {str(e)}"
    
if __name__ == "__main__":
    print(list_ifc_entities())