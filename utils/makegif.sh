#!/bin/bash

if [$# -lt 2]
then
	echo "ディレクトリとdelayを指定してください"
	exit
fi

echo "test $#"
cd $1
DELAY=$2
convert -layers optimize -delay ${DELAY} *.jpeg image.gif

cd ../
