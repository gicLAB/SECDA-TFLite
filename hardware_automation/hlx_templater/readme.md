# HLX Script Templater

For Hardware Automation to work completely, the HLX aspect requires a TCL script with appropriate block design to create an FPGA overlay. We already provide some template TCL scripts that contain certain common block designs. This tool is to create new template TCL scripts for your specialized accelerator IP which might have different ports than any of the previous templates and hence require custom block design.

## Usage
- Create valid block design for your accelerator IP within Vivado 2019.2.
- Make sure to place all blocks required, and to set the address editor as required for the design.
- Validate the design (check mark icon)
- Optional: Run bitstream generation, verify the design can be mapped and works as expected.
- Generate our "src" TCL script from Vivado of your project: 
  - File -> Project -> Write Tcl
  - Save output file in src directory within this folder
  - Tick the following boxes:
    - "Copy sources to new project"
    - "Recreate Block Designs using Tcl"
    - "Ignore command errors"
- This "src" TCL script needs to modified into templated version to do this use the  [hlx_script_templater](hlx_script_templater.ipynb) to generate templated TCL script.
- Make sure to update "tcl_src_file_name" and "ip_name"
- Move the TCL script into [hlx_scripts](../hlx_scripts) folder in the parent directory
- When creating new Hardware Config for your accelerator in the "hlx_tcl_script" field use the template TCL script you just created.
- This will ensure your accelerator IP works during Hardware Automation by using your new templated block design TCL script


## Notes
- One create the template TCL script can be used for new custom accelerator which have the same exactly IP block I/O
- If you are creating new versions of the same architecture without any IP block I/O changes then you can reuse the existing template TCL script for Hardware Automation
- Remember to rerun [hw_gen.py](../hw_gen.py) to ensure your changes affect your next hardware automation run