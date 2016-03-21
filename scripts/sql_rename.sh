#!/usr/bin/env bash

sql_file=$1
src_name=$2
dst_name=$3

tables=$(sqlite3 $sql_file "SELECT name FROM sqlite_master WHERE type='table';")
for table in $tables; do
  cmd="UPDATE $table SET model='$dst_name' WHERE model='$src_name';"
  echo $cmd
  sqlite3 $sql_file "$cmd";
done
