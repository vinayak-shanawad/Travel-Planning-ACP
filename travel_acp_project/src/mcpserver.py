from colorama import Fore
from mcp.server.fastmcp import FastMCP
import json
import requests

mcp = FastMCP("travelagencyserver")

@mcp.tool()
def list_travel_agencies(city: str) -> str:
    """
    This tool returns a list of travel agencies located in the given city.

    Args:
        city (str): The city name to search agencies for. Example: "Hyderabad"

    Returns:
        str: A list of travel agencies matching the city.
        Example: '[{"name": "GlobeTrotters Hyderabad", "rating": 4.6, "contact": "9876543210"}, ...]'
    """
    url = 'https://raw.githubusercontent.com/vinayak-shanawad/Travel-Planning-ACP/main/travel_agencies.json'
    resp = requests.get(url)
    agencies = json.loads(resp.text)

    matched = [a for a in agencies if a['city'].lower() == city.lower()]
    return json.dumps(matched, indent=2)

if __name__ == "__main__":
    mcp.run(transport="stdio")
