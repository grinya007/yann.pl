#!/usr/bin/env perl
use strict;
use warnings;

use FindBin qw/$RealBin/;
use lib "$RealBin/lib";

use AI::YANN::Model::Classification;
use PDL;

my ($n, $in) = (10, 16);

my $model = AI::YANN::Model::Classification->new(
  'layers'  => [
    { 'output_size' => 4, 'input_size' => $in, 'activation' => 'relu' },
    { 'output_size' => 4, 'activation' => 'relu' },
    { 'output_size' => 2, 'activation' => 'softmax' },
  ],
  'lr'      => 0.01,
);

my $epochs = 20;
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
  print $batch->[1], " ", $p, "\n";
}


sub batches {
  my ($n, $m, $s) = @_;
  my @batches;
  for (1 .. $n) {
    my $x = random($s, $m);
    $x->where($x > 0.5) .= 1;
    $x->where($x <= 0.5) .= 0;
    my $y = $x->slice(1)->transpose();
    push(@batches, [$x, $y]);
  }

  return \@batches;
}
