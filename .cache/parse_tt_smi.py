#!/usr/bin/env python3
"""Tiny local helper: extract relevant fields from a tt-smi -s JSON dump."""
import json, sys

p = sys.argv[1] if len(sys.argv) > 1 else "/Users/adityasriram/Labs/stanford/cs440lx/tt-model-bringup/.cache/qb1_tt_smi.json"
with open(p) as f:
    d = json.load(f)

print("host_sw_vers:", d["host_sw_vers"])
print("num devices:", len(d["device_info"]))
print()
for i, di in enumerate(d["device_info"]):
    bi = di["board_info"]
    smb = di["smbus_telem"]
    et_col = int(smb["ENABLED_TENSIX_COL"], 16)
    # ENABLED_TENSIX_COL is a column-mask bitfield
    cols_enabled = bin(et_col).count("1")
    print(f"[{i}] board={bi['board_type']} dram={bi['dram_speed']} dram_ok={bi['dram_status']}")
    print(f"     bus={bi['bus_id']} pcie={bi['pcie_speed']}x{bi['pcie_width']}")
    print(f"     ENABLED_TENSIX_COL = 0x{et_col:x} ({cols_enabled} columns set)")
    print(f"     ENABLED_GDDR       = 0x{int(smb['ENABLED_GDDR'],16):x} ({bin(int(smb['ENABLED_GDDR'],16)).count('1')} channels)")
    print(f"     ENABLED_L2CPU      = 0x{int(smb['ENABLED_L2CPU'],16):x}")
    print(f"     FLASH_BUNDLE_VERSION = 0x{int(smb['FLASH_BUNDLE_VERSION'],16):08x}")
    print(f"     CM_FW_VERSION        = 0x{int(smb['CM_FW_VERSION'],16):08x}")
    print(f"     DM_APP_FW_VERSION    = 0x{int(smb['DM_APP_FW_VERSION'],16):08x}")
    print(f"     GDDR_FW_VERSION      = 0x{int(smb['GDDR_FW_VERSION'],16):08x}")
    print(f"     ARC L2CPU clocks     = {int(smb['L2CPUCLK0'],16)} MHz")
    print(f"     AICLK = {int(smb['AICLK'],16)} MHz  AXICLK = {int(smb['AXICLK'],16)} MHz")
    if "firmwares" in di:
        print("     firmwares:", di["firmwares"])
    if "limits" in di:
        print("     limits:", di["limits"])
    print()
