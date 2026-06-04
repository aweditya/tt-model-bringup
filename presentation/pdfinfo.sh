#!/bin/bash
export PATH="/Library/TeX/texbin:/usr/local/bin:$PATH"
pdfinfo "$1"
