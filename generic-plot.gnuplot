# Set CSV data format
set datafile separator ","

# Set output format to PNG image
set terminal pngcairo enhanced size 800,600

# Set graph title, axis labels
set title "Title Here"
set xlabel "X-axis Label Here"
set ylabel "Validation Accuracy"

# Set gridW
set grid


set output 'batch-size.png'
set title "Validation Accuracy vs. Batch Size"
set xlabel "Batch Size"
plot 'batch_size.csv' using 1:2 with linespoints title 'Accuracy'

set output 'num-hidden-layers.png'
set title "Validation Accuracy vs. Number of Hidden Layers"
set xlabel "Number of Hidden Layers"
plot 'hidden_layers.csv' using 1:2 with linespoints title 'Accuracy'

set output 'learning-rate.png'
set title "Validation Accuracy vs. Learning Rate"
set xlabel "Learning Rate"
plot 'learning_rate.csv' using 1:2 with linespoints title 'Accuracy'

