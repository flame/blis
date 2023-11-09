"""Copyright (C) 2020 - 2023, Advanced Micro Devices, Inc. All rights reserved."""

import subprocess
import sys

def config_check():
    # Execute wmic shell command with sub-process
    result = subprocess.run(['wmic', 'cpu', 'get', 'caption'], stdout=subprocess.PIPE, text=True).stdout

    # Replace the newline character with empty char
    result=result.replace('\n', '')

    # parse the string into list of string
    parse_string=result.split(" ")

    # Strip the empty strings from list
    parse_string=[list for list in parse_string if list.strip()]

    vendor=parse_string[1]
    family=hex(int(parse_string[3]))
    model=hex(int(parse_string[5]))
    stepping=hex(int(parse_string[7]))

    # AMD family numbers
    # Zen/ Zen+/Zen2 family number
    zen_family="0x17"
    # Bulldozer / Piledriver / Steamroller / Excavator family number
    amd_family="0x15"

    # AMD CPUID model numbers
    zen_model=["0x30", "0xff"]
    zen2_model=["0x00", "0xff"]
    excavator_model=["0x60","0x7f"]
    steamroller_model=["0x30", "0x3f"]
    piledriver_model=["0x02", "0x10", "0x1f"]
    bulldozer_model=["0x00", "0x01"]

    # Check the CPU configuration Intel64/AMD64
    if vendor.count("Intel64"):
        return
    elif vendor.count("AMD64"):
        # Check the AMD family name
        if family == zen_family:
            if (zen_model[0] <= model and model <= zen_model[1]) :
                family="zen2"
            elif (zen2_model[0] <= model and model <= zen2_model[1]) :
                family="zen"
            else:
                print("Unknown model number")
        elif family == amd_family:
            # check for specific models of excavator family
            if (excavator_model[0] <= model and model <= excavator_model[1]) :
                family="excavator"
            # check for specific models of steamroller family
            elif (steamroller_model[0] <= model and model <= steamroller_model[1]) :
                family="steamroller"
            # check for specific models of piledriver family
            elif (model == piledriver_model[0] or (piledriver_model[1] <= model and model <= piledriver_model[2])) :
                family="piledriver"
            # check for specific models of bulldozer family
            elif (model == bulldozer_model[0] or model == bulldozer_model[1]) :
                family="bulldozer"
            else:
                print("Unknown model number")
        else:
            print("Unknown family")
    else:
        print("UNKNOWN CPU")
    return family

# Function call for config family names
FAMILY=config_check()
print(FAMILY)
