#!/usr/bin/env python

#
# Patch ld.so to disable runtime CPUID detection
# Taken from https://stackoverflow.com/a/44483482
#

import re
import sys

infile, outfile = sys.argv[1:]
d = open(infile, 'rb').read()
# Match CPUID(eax=0), "xor eax,eax" followed closely by "cpuid"
o = re.sub(b'(\x31\xc0.{0,32})\x0f\xa2', b'\\1\x66\x90', d)
#assert d != o
open(outfile, 'wb').write(o)
