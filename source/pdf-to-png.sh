#!/bin/bash

#
# Script to convert all invoices from PDFs to PNGs
#

find .data/pdf -name '*.pdf' -exec ./pdf-to-png-single.sh {} \;