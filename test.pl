#!/usr/bin/env perl
use strict;
use warnings;

use FindBin qw/$RealBin/;
use lib "$RealBin/lib";

use AI::YANN::Model::Regression;
use PDL;

my ($n, $in) = (6, 5);

my $model = AI::YANN::Model::Regression->new(
  'layers'  => [
    { 'output_size' => 3, 'input_size' => $in, 'activation' => 'relu' },
    #{ 'output_size' => 64, 'activation' => 'relu' },
    { 'output_size' => 1 },
  ],
  'lr'      => 0.001,
);

my $epochs = 1;
my $batches = batches(1, $n, $in);
for my $epoch (1 .. $epochs) {
  my @losses;
  for my $batch (@$batches) {
    my $loss = $model->fit(@$batch);
    push(@losses, $loss);
  }
  print "$epoch/$epochs ", pdl(\@losses)->avg(), "\n";
}


#for my $batch (@{ batches(4, $n, $in) }) {
  #my $p = $model->predict($batch->[0]);
  #print $batch->[1]->transpose(), $p, "\n";
#}


sub batches {
  my ($n, $m, $s) = @_;
  my @batches;
  for (1 .. $n) {
    my $x = random($s, $m);

    my $y = $x->copy();
    #$y->slice(0) *= 7;
    #$y->slice(1) += ($y->slice(0) + $y->slice(1) * 10) ** (1 + $y->slice(2));
    #$y->slice(2) *= 5 * $y->slice(6);
    #$y->slice(3) += ($y->slice(2) + $y->slice(7) * 10) ** (1 + $y->slice(5));
    $y = $y->sumover()->transpose();

    push(@batches, [$x, $y]);
  }

  return \@batches;
}
