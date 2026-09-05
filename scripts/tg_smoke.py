from agrag.graph.client import connect

if __name__ == "__main__":
    conn = connect()
    print("version:", conn.getVer())
    print("vertex types:", conn.getVertexTypes())
    print("Event count:", conn.getVertexCount("Event"))
