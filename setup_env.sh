#!/bin/sh

if [ "$(basename -- "$0")" == "$(basename -- "$BASH_SOURCE")" ]; then
  echo "Don't run $0, source it" >&2
  echo "source `basename $BASH_SOURCE` in directory $dir"
  exit 1
fi

if [ "$(dirname -- "$(realpath $BASH_SOURCE)")" != "$PWD" ]; then
  echo "FAIL: source `basename $BASH_SOURCE` in directory $dir"
  return
fi

export SIMHOME=$PWD
export BOOKSIMSRC=$PWD/src/booksim2/src

echo "source successfully!"
