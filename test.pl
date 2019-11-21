#!/usr/bin/env perl
use strict;
use warnings;

use FindBin qw/$RealBin/;
use lib "$RealBin/lib";

use AI::YANN::Model::Regression;
use PDL;

my ($n, $in) = (5, 8);

my $model = AI::YANN::Model::Regression->new(
  'layers'  => [
    { 'output_size' => 64, 'input_size' => $in, 'activation' => 'sigmoid' },
    #{ 'output_size' => 32, 'activation' => 'relu' },
    #{ 'output_size' => 32, 'activation' => 'relu' },
    { 'output_size' => 1 },
  ],
  'lr'      => 0.1,
);

my $epochs = 500;
my $batches = batches(20, $n, $in);
for my $epoch (1 .. $epochs) {
  my @losses;
  for my $batch (@$batches) {
    my $loss = $model->fit(@$batch);
    push(@losses, $loss);
  }
  print "$epoch/$epochs ", pdl(\@losses)->avg(), "\n";
}


for my $batch (@{ batches(1, $n, $in) }) {
  my $p = $model->predict($batch->[0]);
  print $batch->[1], $p, "\n";
}


sub batches {
  my ($n, $m, $s) = @_;
  my @batches;
  for (1 .. $n) {
    my $x = random($s, $m);

    my $y = $x->copy();
    $y->slice(0) *= 7;
    $y->slice(1) += ($y->slice(0) + $y->slice(1) * 10) ** (1 + $y->slice(2));
    $y->slice(2) += ($y->slice(0) + $y->slice(2) * 10) ** (1 + $y->slice(5));
    $y->slice(3) *= 5 * $y->slice(4);
    $y = $y->sumover()->transpose();

    push(@batches, [$x, $y]);
  }

  return \@batches;
}
