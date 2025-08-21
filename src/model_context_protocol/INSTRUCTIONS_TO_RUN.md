# Instructions to run MCP Demo 

## 1. Rubik PI
  * SSH to Device (recommend using VSCode from Snapdragon for automatic tunneling)
  * Setup virtual environment.
  * Activate Environment:
  ```bash
  source {name_of_virtual_environment}/bin/activate
  pip install -r rubik-pi-mcp-server/requirements.txt
  ```
  * Start server:
  ```bash
  python rubik-pi-mcp-server/main.py
  ```
## 2. Snapdragon X Elite 
  * Setup virtual environment.
  * Activate Environment:
 ```powershell
 {name_of_virtual_environment}/Scripts/activate.ps1
 pip install -r requirements.txt
 ```
  * Start client: 
 ```powershell
 python client_host/mcp_client.py
 ```
  * If you want access to HIGH security (local inference), git clone qnn_sample_apps and all dependencies for gemma3-1B
