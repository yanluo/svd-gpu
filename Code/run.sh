#i/bin/sh

rm -f result.txt

for ((i=15000; i<=17000; i=i+1000))
do
#  for((j=1; j<=10; j++))
#  do
    ./caffe -n $i -i 1 -p 10e-8 > $i.txt
#  done
done

