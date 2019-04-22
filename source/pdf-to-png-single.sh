#!/bin/bash

#
# Script to convert an invoice from PDF to PNG
#

if [[ -f "$1.png" ]]; then
  echo "Skipping $1 as it already exists"
  exit 0;
fi

PAGE_COUNT="$(pdfinfo "$1" | grep Pages: | awk '{print $2}')"

if [[ "$PAGE_COUNT" != "1" ]]; then
  echo "$1 has $PAGE_COUNT pages"
  exit 1
fi

echo "Converting $1 ..."
convert -density 300 "$1" "$1.png"
echo "done"
