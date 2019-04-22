#!/bin/bash

#
# Script to run tesseract OCR on all classified invoices
#

find .data/classified -type f -name '*.png' | xargs -L 1 -J {} -P 5 ./ocr-single.sh {}
