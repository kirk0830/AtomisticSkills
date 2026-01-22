# Structural Visualization with Matterviz

For high-fidelity 3D visualization of crystal structures (CIF, POSCAR, XYZ), we recommend using the **Matterviz** extension. This allows for interactive rendering directly within the VS Code / Antigravity editor. Matterviz visualization works seamlessly even when the structure files and the Antigravity session are on a **remote server** (connected via SSH). The rendering is handled locally in the agent session.

## Installation

### In VS Code
1.  Open the **Extensions** view (Cmd/Ctrl+Shift+X).
2.  Search for **Matterviz**.
3.  Click **Install**.

### In Antigravity
The extension can be installed via the marketplace search within the Antigravity instance.

## Usage
1.  **Locate a structure file** (e.g., `.cif`, `POSCAR`, `.xyz`) in the file explorer.
2.  **Right-click** on the file.
3.  Select **"Render"** or **"Open Matterviz Preview"** from the context menu.
4.  An interactive 3D view will open in a new tab, allowing you to:
    - **Rotate**: Left-click and drag.
    - **Zoom**: Scroll wheel.
    - **Pan**: Right-click and drag.

