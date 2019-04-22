#!/bin/bash

#
# Script to run tesseract OCR on an invoice
#

USE_TSV=no # yes

FILE_SUFFIX=txt
TESS_CONF=

if [[ "$USE_TSV" == "yes" ]]; then
    FILE_SUFFIX=tsv
    TESS_CONF=tsv
fi

echo "Running tesseract on $1..."

if [[ ! -f "$(pwd)/$1" ]]; then
  echo "File $(pwd)/$1 doesn't exist";
  exit 1;
fi

if [[ -f "$(pwd)/$1.$FILE_SUFFIX" ]]; then
  echo "Already done..."
  exit 0;
fi

docker run --rm -v $(pwd):/workdir --env 'TESSDATA_PREFIX=/tesstrain/tessdata_best' sventschui/ocr-lab:latest tesseract -l deu /workdir/$1 /workdir/$1 $TESS_CONF

exit $?
