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
if o == d:
  # Match CPUID(eax=1), "cpuid" followed closely by "and ecx,18000000h"
  o = re.sub(b'(\x0f\xa2.{0,32})\x81\xe1\x00\x00\x00\x18', b'\\1\x81\xe1\x00\x00\x00\x00', d)
  assert d != o
open(outfile, 'wb').write(o)
