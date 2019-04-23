#!/bin/bash

#
# Script to query local copy of the AXA invoices DB to classify all invoices
#

find .data/pdf -name '*.pdf' -exec ./classify-single.sh {} \;

