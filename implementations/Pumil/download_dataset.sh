#!/bin/sh

usage() {
    cat 1>&2 << EOF
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
OVERVIEW: download and uncompress datasets from UCI repository.
  
USAGE:
      $(basename ${0}) [dataset]
  
DATASET:
 musk1          drug activity (version 1)
 musk2          drug activity (version 2)
 elephant       Corel Image Set (Elephant)
 fox            Corel Image Set (Fox)
 tiger          Corel Image Set (Tiger)
 tst1           TREC9 / 1
 tst2           TREC9 / 2
 tst3           TREC9 / 3
 tst4           TREC9 / 4
 tst7           TREC9 / 7
 tst9           TREC9 / 9
 tst10          TREC9 / 10
-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-
EOF
exit -1
}

dl_trec9() {
    if [ ! -d MilData ]; then
      curl -s -O http://www.cs.columbia.edu/~andrews/mil/data/MIL-Data-2002-Musk-Corel-Trec9.tgz
      tar xf MIL-Data-2002-Musk-Corel-Trec9.tgz
    fi

    if [ "$1" = "elephant" ]; then
      mv MilData/Elephant/data_100x100.svm elephant.data
    elif [ "$1" = "fox" ]; then
      mv MilData/Fox/data_100x100.svm      fox.data
    elif [ "$1" = "tiger" ]; then
      mv MilData/Tiger/data_100x100.svm    tiger.data
    elif [ "$1" = "musk1" ]; then
      mv MilData/Musk/musk1norm.svm        musk1.data
    elif [ "$1" = "musk2" ]; then
      mv MilData/Musk/musk2norm.svm        musk2.data
    elif [ "$1" = "tst1" ]; then
      mv MilData/1/data_200x200.svm        tst1.data
    elif [ "$1" = "tst2" ]; then
      mv MilData/2/data_200x200.svm        tst2.data
    elif [ "$1" = "tst3" ]; then
      mv MilData/3/data_200x200.svm        tst3.data
    elif [ "$1" = "tst4" ]; then
      mv MilData/4/data_200x200.svm        tst4.data
    elif [ "$1" = "tst7" ]; then
      mv MilData/7/data_200x200.svm        tst7.data
    elif [ "$1" = "tst9" ]; then
      mv MilData/9/data_200x200.svm        tst9.data
    elif [ "$1" = "tst10" ]; then
      mv MilData/10/data_200x200.svm       tst10.data
    fi
}

if [ "${1}" = "" ]; then
    usage
fi

case ${1} in
musk1)
    dl_trec9 musk1;;
musk2)
    dl_trec9 musk2;;
elephant)
    dl_trec9 elephant;;
fox)
    dl_trec9 fox;;
tiger)
    dl_trec9 tiger;;
tst1)
    dl_trec9 tst1;;
tst2)
    dl_trec9 tst2;;
tst3)
    dl_trec9 tst3;;
tst4)
    dl_trec9 tst4;;
tst7)
    dl_trec9 tst7;;
tst9)
    dl_trec9 tst9;;
tst10)
    dl_trec9 tst10;;
--help | -h)
    usage;;
*)
    echo "error: Unknown dataset name '${1}'" 1>&2
    usage;;
esac
