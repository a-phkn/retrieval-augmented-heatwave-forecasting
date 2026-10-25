#!/bin/bash
cd "$HOME/mnt/retrieval-augmented-heatwave-forecasting/sources"
for i in $(seq 1 62); do
  n=$(printf "%02d" $i)
  [ -s ocr_ndma2019/p$n.txt ] && continue
  pdftoppm -r 200 -f $i -l $i -png pdfs/NDMA_HW_GUIDELINES_2019.pdf /tmp/ndma_p$n >/dev/null 2>&1
  f=$(ls /tmp/ndma_p$n*.png | head -1)
  tesseract "$f" ocr_ndma2019/p$n -l eng >/dev/null 2>&1
  rm -f /tmp/ndma_p$n*.png
done
echo done > ocr_ndma2019/DONE
