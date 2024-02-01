set term png
set datafile separator ","
set output "filename.png"
set xlabel "X axis label"
set ylabel "Y axis label"
plot "filename.csv" using 1:2 title "Plot title" with lines
