#!/bin/bash
mkdir -p designs
for i in {1..10}
do
  url="https://www.ispd.cc/contests/19/benchmarks/ispd19_test$i.tgz";
  echo $url;
  (wget -P designs $url && gunzip designs/ispd19_test$i.tgz && tar -xvf designs/ispd19_test$i.tar -C designs) &
  rm -rf designs/ispd19_test$i.tar
done
