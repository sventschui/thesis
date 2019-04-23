#!/bin/bash

#
# Script to query local copy of the AXA invoices DB to classify an invoice
#

FILENAME=$(basename -- "$1")
DOC_ID=${FILENAME%.*}

MONGO_QUERY="db.getCollection('eclaimInvoice').aggregate([{ \$match:{'documents.0.documentStorageId':'$DOC_ID'}},{\$lookup:{ from: 'zsrClassMapping', as: 'class', localField: 'serviceProvider', foreignField:'_id'}},{\$unwind:'\$class'},{\$project: { 'class': '\$class.class'}}])"
RESULT="$(docker run -it --rm --net host mongo:3.4 mongo --eval "$MONGO_QUERY" --quiet classification)"

if [ -z "$RESULT" ]; then
  CLASS="unknown"
else
  if [ "$(echo "$RESULT" | wc -l | tr -d '[:space:]')" != "1" ]; then
    echo "Could not classify $1 as there is more than one result..."
    exit 1
  fi

  JQ_RES="$(echo "$RESULT" | jq -cre '.class')"

  if [ "$JQ_RES" == "null" ]; then
    CLASS="unknown"
  else
    CLASS="$JQ_RES"
  fi
fi 

mkdir -p ".data/classified/$CLASS"
ln -s "$(pwd)/$1" ".data/classified/$CLASS/$FILENAME"


